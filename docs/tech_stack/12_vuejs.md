# 12. Vue.js: Vue 3, Composition API & Component Architecture

> **Context**: Vue.js is listed as a bonus skill for a backend AI engineer role. This document
> covers the essentials you need to demonstrate solid front-end competence without the depth
> expected of a dedicated front-end engineer.

---

## Table of Contents

1. [Vue.js Overview](#1-vuejs-overview)
2. [Vue 3 Composition API](#2-vue-3-composition-api)
3. [Component Architecture](#3-component-architecture)
4. [Reactivity System](#4-reactivity-system)
5. [Composables (Custom Hooks)](#5-composables-custom-hooks)
6. [State Management with Pinia](#6-state-management-with-pinia)
7. [Vue Router](#7-vue-router)
8. [Template Syntax & Directives](#8-template-syntax--directives)
9. [Building an AI-Powered Vue Frontend](#9-building-an-ai-powered-vue-frontend)
10. [Vite & Build Tools](#10-vite--build-tools)
11. [Q&A Section (20 Questions)](#11-qa-section)

---

## 1. Vue.js Overview

### What Is Vue.js?

Vue.js is a **progressive JavaScript framework** for building user interfaces. "Progressive" means
you can adopt it incrementally: use it as a simple script tag for one widget, or scale it into a
full single-page application with routing, state management, and server-side rendering.

**Core philosophy**: an approachable, performant, and versatile framework that does not force
architectural decisions on you up front.

### Vue 2 vs Vue 3 -- Key Differences

| Aspect | Vue 2 | Vue 3 |
|---|---|---|
| **Reactivity engine** | `Object.defineProperty` | ES6 `Proxy` (faster, catches more mutations) |
| **API style** | Options API only | Options API + **Composition API** |
| **Tree-shaking** | Limited | Full (unused features are excluded from bundle) |
| **Performance** | Good | ~2x faster virtual DOM, smaller bundle |
| **TypeScript** | Partial support | First-class TypeScript support |
| **Fragments** | Single root element required | Multiple root elements allowed |
| **Teleport** | Not available | `<Teleport>` for rendering outside component tree |
| **Suspense** | Not available | `<Suspense>` for async dependencies |
| **State management** | Vuex | **Pinia** (official) |
| **Build tool** | Vue CLI (webpack) | **Vite** (esbuild + rollup) |

### Options API vs Composition API

**Options API** organizes code by *option type* (data, methods, computed, watch, lifecycle hooks).
This works well for small components but leads to "scattered logic" when a component handles
multiple concerns.

```javascript
// Options API
export default {
  data() {
    return { count: 0 }
  },
  computed: {
    doubled() { return this.count * 2 }
  },
  methods: {
    increment() { this.count++ }
  },
  mounted() {
    console.log('mounted')
  }
}
```

**Composition API** organizes code by *logical concern*. Related state, computed values, watchers,
and functions live together, making it easy to extract into reusable **composables**.

```vue
<script setup>
import { ref, computed, onMounted } from 'vue'

const count = ref(0)
const doubled = computed(() => count.value * 2)
function increment() { count.value++ }
onMounted(() => console.log('mounted'))
</script>
```

**When to use which**: Composition API is the recommended default in Vue 3. Options API is still
fully supported and works fine for simple components, but Composition API scales better and
provides superior TypeScript integration.

### Vue Ecosystem

| Tool | Purpose |
|---|---|
| **Vue Router** | Official client-side routing |
| **Pinia** | Official state management (replaced Vuex) |
| **Vite** | Fast build tool using native ES modules |
| **Nuxt 3** | Full-stack Vue framework (SSR, SSG, file-based routing) |
| **VueUse** | Collection of 200+ ready-made composables |
| **Vitest** | Vite-native unit testing framework |
| **Vue DevTools** | Browser extension for debugging |

---

## 2. Vue 3 Composition API

### `<script setup>` -- The Modern Syntax

`<script setup>` is a compile-time syntactic sugar that:
- Automatically exposes all top-level bindings to the template
- Removes boilerplate (`export default`, `setup()` return)
- Enables better TypeScript inference
- Produces more efficient compiled code

```vue
<script setup>
// Everything declared here is available in <template>
import { ref } from 'vue'
import MyComponent from './MyComponent.vue'

const message = ref('Hello')
</script>

<template>
  <MyComponent />
  <p>{{ message }}</p>
</template>
```

Without `<script setup>`, the equivalent is:

```javascript
import { ref } from 'vue'
import MyComponent from './MyComponent.vue'

export default {
  components: { MyComponent },
  setup() {
    const message = ref('Hello')
    return { message }
  }
}
```

### `ref()` vs `reactive()`

Both create reactive state. The key difference is **how** they wrap the value.

#### `ref()` -- For Any Value

Wraps the value in a `{ value: ... }` object. Works with primitives and objects.

```javascript
import { ref } from 'vue'

const count = ref(0)          // Ref<number>
const user = ref({ name: 'Alice' })  // Ref<{ name: string }>

// Access in JS requires .value
count.value++
console.log(user.value.name)

// In templates, .value is automatically unwrapped
// <p>{{ count }}</p>  -- no .value needed
```

#### `reactive()` -- For Objects Only

Returns a Proxy of the original object. Cannot hold primitives.

```javascript
import { reactive } from 'vue'

const state = reactive({
  count: 0,
  user: { name: 'Alice' }
})

// No .value needed
state.count++
console.log(state.user.name)
```

**Rule of thumb**: prefer `ref()` for everything. It is more consistent, works with primitives,
and avoids the destructuring pitfall of `reactive()` (see Section 4).

### `computed()`

Creates a cached, read-only reactive value derived from other reactive state.

```javascript
import { ref, computed } from 'vue'

const items = ref([
  { name: 'Apple', price: 1.5 },
  { name: 'Banana', price: 0.8 },
  { name: 'Cherry', price: 3.0 }
])

// Automatically recalculates when items change
const total = computed(() =>
  items.value.reduce((sum, item) => sum + item.price, 0)
)

const expensive = computed(() =>
  items.value.filter(item => item.price > 1)
)
```

**Writable computed** (rare, but exists):

```javascript
const firstName = ref('John')
const lastName = ref('Doe')

const fullName = computed({
  get: () => `${firstName.value} ${lastName.value}`,
  set: (val) => {
    const [first, last] = val.split(' ')
    firstName.value = first
    lastName.value = last
  }
})

fullName.value = 'Jane Smith' // sets firstName='Jane', lastName='Smith'
```

### `watch()` and `watchEffect()`

#### `watch()` -- Explicit Source Watching

Watches one or more specific reactive sources. Provides old and new values. Lazy by default (does
not run immediately).

```javascript
import { ref, watch } from 'vue'

const searchQuery = ref('')

// Watch a single ref
watch(searchQuery, (newVal, oldVal) => {
  console.log(`Search changed: "${oldVal}" -> "${newVal}"`)
  fetchResults(newVal)
})

// Watch multiple sources
const page = ref(1)
watch([searchQuery, page], ([newQuery, newPage], [oldQuery, oldPage]) => {
  fetchResults(newQuery, newPage)
})

// Immediate execution (run on mount too)
watch(searchQuery, (val) => {
  fetchResults(val)
}, { immediate: true })

// Deep watching for nested objects
const filters = ref({ category: 'all', sort: 'date' })
watch(filters, (newFilters) => {
  applyFilters(newFilters)
}, { deep: true })
```

#### `watchEffect()` -- Automatic Dependency Tracking

Runs immediately, automatically tracks every reactive dependency accessed inside the callback.

```javascript
import { ref, watchEffect } from 'vue'

const url = ref('/api/items')
const page = ref(1)

// Automatically re-runs when url or page changes
watchEffect(async () => {
  const response = await fetch(`${url.value}?page=${page.value}`)
  const data = await response.json()
  console.log(data)
})
```

**`watch` vs `watchEffect`**:
- Use `watch` when you need old/new values or want lazy execution
- Use `watchEffect` when you want automatic dependency tracking and immediate execution

### Lifecycle Hooks (Composition API)

| Options API | Composition API | When it fires |
|---|---|---|
| `beforeCreate` | Not needed (use `setup`) | Before instance initialization |
| `created` | Not needed (use `setup`) | After instance initialization |
| `beforeMount` | `onBeforeMount` | Before DOM mount |
| `mounted` | `onMounted` | After DOM mount |
| `beforeUpdate` | `onBeforeUpdate` | Before DOM re-render |
| `updated` | `onUpdated` | After DOM re-render |
| `beforeUnmount` | `onBeforeUnmount` | Before component is destroyed |
| `unmounted` | `onUnmounted` | After component is destroyed |
| -- | `onErrorCaptured` | When child component error is captured |

```vue
<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const windowWidth = ref(window.innerWidth)

function handleResize() {
  windowWidth.value = window.innerWidth
}

onMounted(() => {
  window.addEventListener('resize', handleResize)
  console.log('Component mounted, DOM is ready')
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  console.log('Cleanup complete')
})
</script>

<template>
  <p>Window width: {{ windowWidth }}px</p>
</template>
```

### Comprehensive Composition API Example

```vue
<script setup>
import { ref, computed, watch, onMounted } from 'vue'

// --- Reactive state ---
const count = ref(0)
const message = ref('')

// --- Computed ---
const doubled = computed(() => count.value * 2)
const isEven = computed(() => count.value % 2 === 0)

// --- Methods ---
function increment() {
  count.value++
}

function reset() {
  count.value = 0
  message.value = ''
}

// --- Watchers ---
watch(count, (newVal) => {
  if (newVal >= 10) {
    message.value = 'Count reached 10!'
  }
})

// --- Lifecycle ---
onMounted(() => {
  console.log('Counter component is ready')
})
</script>

<template>
  <div class="counter">
    <button @click="increment">
      Count: {{ count }} (doubled: {{ doubled }})
    </button>
    <p v-if="isEven">The count is even</p>
    <p v-else>The count is odd</p>
    <p v-if="message" class="alert">{{ message }}</p>
    <button @click="reset">Reset</button>
  </div>
</template>

<style scoped>
.counter {
  padding: 1rem;
}
.alert {
  color: red;
  font-weight: bold;
}
</style>
```

---

## 3. Component Architecture

### Single File Components (SFCs)

A `.vue` file encapsulates template, logic, and styles in one file:

```vue
<script setup>
// JavaScript / TypeScript logic
</script>

<template>
  <!-- HTML template -->
</template>

<style scoped>
/* CSS scoped to this component */
</style>
```

`scoped` styles are compiled to use unique attribute selectors so they only affect the current
component. Without `scoped`, styles are global.

### Props

Props are the primary way to pass data **from parent to child**.

```vue
<!-- ChildComponent.vue -->
<script setup>
// defineProps is a compiler macro -- no import needed
const props = defineProps({
  title: {
    type: String,
    required: true
  },
  count: {
    type: Number,
    default: 0
  },
  items: {
    type: Array,
    default: () => []
  }
})

// TypeScript syntax (preferred in TS projects)
// const props = defineProps<{
//   title: string
//   count?: number
//   items?: string[]
// }>()
</script>

<template>
  <h2>{{ title }}</h2>
  <p>Count: {{ count }}</p>
  <ul>
    <li v-for="item in items" :key="item">{{ item }}</li>
  </ul>
</template>
```

```vue
<!-- ParentComponent.vue -->
<template>
  <ChildComponent
    title="My List"
    :count="42"
    :items="['apple', 'banana', 'cherry']"
  />
</template>
```

### Emits

Emits are the primary way to communicate **from child to parent**.

```vue
<!-- SearchInput.vue -->
<script setup>
const emit = defineEmits(['search', 'clear'])

// TypeScript syntax
// const emit = defineEmits<{
//   (e: 'search', query: string): void
//   (e: 'clear'): void
// }>()

function handleSearch(event) {
  emit('search', event.target.value)
}

function handleClear() {
  emit('clear')
}
</script>

<template>
  <div>
    <input @input="handleSearch" placeholder="Search..." />
    <button @click="handleClear">Clear</button>
  </div>
</template>
```

```vue
<!-- Parent.vue -->
<script setup>
import { ref } from 'vue'
import SearchInput from './SearchInput.vue'

const results = ref([])

function onSearch(query) {
  console.log('Searching for:', query)
  // fetch results...
}

function onClear() {
  results.value = []
}
</script>

<template>
  <SearchInput @search="onSearch" @clear="onClear" />
</template>
```

### Slots

Slots allow parent components to inject content into child component templates.

#### Default Slot

```vue
<!-- Card.vue -->
<template>
  <div class="card">
    <slot>Default content if nothing is provided</slot>
  </div>
</template>
```

```vue
<!-- Usage -->
<Card>
  <p>This content goes into the default slot</p>
</Card>
```

#### Named Slots

```vue
<!-- Layout.vue -->
<template>
  <div class="layout">
    <header>
      <slot name="header">Default header</slot>
    </header>
    <main>
      <slot>Default main content</slot>
    </main>
    <footer>
      <slot name="footer">Default footer</slot>
    </footer>
  </div>
</template>
```

```vue
<!-- Usage -->
<Layout>
  <template #header>
    <h1>My App</h1>
  </template>

  <p>Main page content goes here</p>

  <template #footer>
    <p>Copyright 2025</p>
  </template>
</Layout>
```

#### Scoped Slots

The child exposes data that the parent can use when rendering the slot content.

```vue
<!-- ItemList.vue -->
<script setup>
defineProps({ items: Array })
</script>

<template>
  <ul>
    <li v-for="(item, index) in items" :key="item.id">
      <!-- Expose item and index to the parent -->
      <slot :item="item" :index="index">
        {{ item.name }}
      </slot>
    </li>
  </ul>
</template>
```

```vue
<!-- Usage -->
<ItemList :items="products">
  <template #default="{ item, index }">
    <span>{{ index + 1 }}. {{ item.name }} - ${{ item.price }}</span>
  </template>
</ItemList>
```

### Component Communication Patterns Summary

| Pattern | Direction | Use Case |
|---|---|---|
| **Props** | Parent -> Child | Pass data down |
| **Emits** | Child -> Parent | Notify parent of events |
| **v-model** | Two-way | Form-like bindings |
| **Provide/Inject** | Ancestor -> Descendant | Skip intermediate components |
| **Pinia store** | Any -> Any | Global shared state |
| **Event bus** | Any -> Any | Discouraged in Vue 3 |

### Provide / Inject

Used to share data from an ancestor to any deeply nested descendant without prop drilling.

```vue
<!-- GrandParent.vue -->
<script setup>
import { provide, ref } from 'vue'

const theme = ref('dark')
const toggleTheme = () => {
  theme.value = theme.value === 'dark' ? 'light' : 'dark'
}

// Any descendant can inject these
provide('theme', theme)
provide('toggleTheme', toggleTheme)
</script>
```

```vue
<!-- DeeplyNestedChild.vue -->
<script setup>
import { inject } from 'vue'

const theme = inject('theme', 'light')          // 'light' is fallback default
const toggleTheme = inject('toggleTheme', () => {})
</script>

<template>
  <div :class="theme">
    <button @click="toggleTheme">Toggle Theme</button>
  </div>
</template>
```

### v-model on Components

`v-model` provides two-way binding. In Vue 3 it uses `modelValue` prop + `update:modelValue` emit.

```vue
<!-- CustomInput.vue -->
<script setup>
defineProps({ modelValue: String })
const emit = defineEmits(['update:modelValue'])
</script>

<template>
  <input
    :value="modelValue"
    @input="emit('update:modelValue', $event.target.value)"
  />
</template>
```

```vue
<!-- Parent.vue -->
<script setup>
import { ref } from 'vue'
import CustomInput from './CustomInput.vue'

const name = ref('')
</script>

<template>
  <!-- These two are equivalent -->
  <CustomInput v-model="name" />
  <CustomInput :modelValue="name" @update:modelValue="name = $event" />
</template>
```

**Named v-model** (multiple v-model bindings):

```vue
<UserForm v-model:first-name="first" v-model:last-name="last" />
```

---

## 4. Reactivity System

### How Vue 3 Reactivity Works

Vue 3 uses **ES6 Proxy** to intercept property access and mutation on reactive objects.

```
Simplified flow:
1. Component renders -> accesses reactive data -> Proxy traps the GET
2. Vue records which component depends on which property (dependency tracking)
3. Data mutates -> Proxy traps the SET
4. Vue notifies all components that depend on that property (trigger)
5. Components re-render with new data
```

This is a significant improvement over Vue 2's `Object.defineProperty` approach:
- **Detects new property additions** (Vue 2 required `Vue.set()`)
- **Detects property deletions**
- **Detects array index mutations** (Vue 2 could not detect `arr[0] = newVal`)
- **Better performance** for large objects

### ref vs reactive -- Deep Dive

```javascript
import { ref, reactive, isRef, isReactive } from 'vue'

// --- ref ---
const count = ref(0)
console.log(isRef(count))        // true
console.log(count.value)          // 0

// ref wrapping an object: the inner object is reactive
const user = ref({ name: 'Alice', age: 30 })
console.log(isRef(user))          // true
console.log(isReactive(user.value)) // true
user.value.name = 'Bob'           // reactive, triggers updates

// --- reactive ---
const state = reactive({ count: 0, name: 'Alice' })
console.log(isReactive(state))    // true
state.count++                      // triggers updates, no .value needed
```

### toRef and toRefs

These create refs that stay connected to a reactive source object.

```javascript
import { reactive, toRef, toRefs } from 'vue'

const state = reactive({
  count: 0,
  name: 'Alice'
})

// toRef: create a ref linked to a single property
const countRef = toRef(state, 'count')
countRef.value++       // also updates state.count

// toRefs: create refs for all properties (great for destructuring)
const { count, name } = toRefs(state)
count.value++           // also updates state.count
name.value = 'Bob'      // also updates state.name
```

**Primary use case**: safely destructure reactive objects (e.g., props) without losing reactivity.

```vue
<script setup>
import { toRefs } from 'vue'

const props = defineProps({ title: String, count: Number })

// Without toRefs, destructuring loses reactivity
// const { title, count } = props  // BAD - not reactive

// With toRefs, refs stay connected
const { title, count } = toRefs(props)
</script>
```

### shallowRef and shallowReactive

Only the top-level is reactive. Nested objects are **not** tracked.

```javascript
import { shallowRef, shallowReactive, triggerRef } from 'vue'

// shallowRef: only .value replacement triggers updates
const data = shallowRef({ nested: { count: 0 } })
data.value.nested.count++        // Does NOT trigger re-render
data.value = { nested: { count: 1 } }  // Triggers re-render (new .value)
triggerRef(data)                  // Force trigger manually

// shallowReactive: only top-level properties are reactive
const state = shallowReactive({
  topLevel: 'reactive',
  nested: { deep: 'not reactive' }
})
state.topLevel = 'new value'     // Triggers re-render
state.nested.deep = 'changed'    // Does NOT trigger re-render
```

**Use case**: performance optimization with large data structures (e.g., a big array from an API
response where you replace the whole thing rather than mutating individual items).

### Reactivity Gotchas

**1. Destructuring reactive objects breaks reactivity**

```javascript
const state = reactive({ count: 0 })
let { count } = state  // count is now a plain number, not reactive!
count++                  // does NOT update state.count

// Fix: use toRefs
const { count } = toRefs(state)
count.value++            // works correctly
```

**2. Replacing a reactive object loses the connection**

```javascript
let state = reactive({ count: 0 })
// This creates a new Proxy, the template still points to the old one
state = reactive({ count: 1 })  // BAD

// Fix: mutate properties instead, or use ref
const state = ref({ count: 0 })
state.value = { count: 1 }  // works
```

**3. Reactive only works with objects**

```javascript
// reactive(0)   -- ERROR: value cannot be made reactive
// reactive('')  -- ERROR: value cannot be made reactive

// Fix: use ref for primitives
const count = ref(0)
```

**4. Forgetting `.value` in JavaScript code**

```javascript
const count = ref(0)
count++        // Wrong! Increments the Ref object itself
count.value++  // Correct

// Note: in templates, .value is auto-unwrapped
// <p>{{ count }}</p>  -- this is fine
```

---

## 5. Composables (Custom Hooks)

### What Are Composables?

Composables are functions that encapsulate **stateful logic** using Composition API. They are
Vue's equivalent of React's custom hooks. By convention, composable function names start with
`use`.

### useFetch -- Data Fetching Composable

```javascript
// composables/useFetch.js
import { ref, watchEffect, toValue } from 'vue'

export function useFetch(url) {
  const data = ref(null)
  const error = ref(null)
  const loading = ref(false)

  async function fetchData() {
    loading.value = true
    error.value = null

    try {
      const resolvedUrl = toValue(url) // supports ref or plain string
      const response = await fetch(resolvedUrl)

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      data.value = await response.json()
    } catch (err) {
      error.value = err
      data.value = null
    } finally {
      loading.value = false
    }
  }

  // If url is a ref, re-fetch when it changes
  watchEffect(() => {
    fetchData()
  })

  return { data, error, loading, retry: fetchData }
}
```

```vue
<!-- Usage -->
<script setup>
import { ref } from 'vue'
import { useFetch } from '@/composables/useFetch'

const endpoint = ref('/api/users')
const { data, error, loading, retry } = useFetch(endpoint)
</script>

<template>
  <div v-if="loading">Loading...</div>
  <div v-else-if="error">Error: {{ error.message }} <button @click="retry">Retry</button></div>
  <ul v-else>
    <li v-for="user in data" :key="user.id">{{ user.name }}</li>
  </ul>
</template>
```

### useLocalStorage -- Persistent State Composable

```javascript
// composables/useLocalStorage.js
import { ref, watch } from 'vue'

export function useLocalStorage(key, defaultValue) {
  const stored = localStorage.getItem(key)
  const data = ref(stored ? JSON.parse(stored) : defaultValue)

  watch(data, (newValue) => {
    localStorage.setItem(key, JSON.stringify(newValue))
  }, { deep: true })

  return data
}
```

```vue
<script setup>
import { useLocalStorage } from '@/composables/useLocalStorage'

const theme = useLocalStorage('theme', 'light')
const favorites = useLocalStorage('favorites', [])
</script>
```

### useDebounce -- Debounced Ref Composable

```javascript
// composables/useDebounce.js
import { ref, watch } from 'vue'

export function useDebounce(source, delay = 300) {
  const debounced = ref(source.value)
  let timeout

  watch(source, (newVal) => {
    clearTimeout(timeout)
    timeout = setTimeout(() => {
      debounced.value = newVal
    }, delay)
  })

  return debounced
}
```

```vue
<script setup>
import { ref, watch } from 'vue'
import { useDebounce } from '@/composables/useDebounce'
import { useFetch } from '@/composables/useFetch'

const searchInput = ref('')
const debouncedSearch = useDebounce(searchInput, 500)

// Only fetches after user stops typing for 500ms
const { data } = useFetch(
  computed(() => `/api/search?q=${debouncedSearch.value}`)
)
</script>

<template>
  <input v-model="searchInput" placeholder="Search..." />
</template>
```

### Composable Best Practices

1. **Naming**: always prefix with `use` (`useFetch`, `useAuth`, `useTheme`)
2. **Return refs**: return `ref` values so the caller can destructure without losing reactivity
3. **Accept ref or plain value**: use `toValue()` to normalize inputs
4. **Handle cleanup**: use `onUnmounted` or the `watch` cleanup callback for event listeners, timers, etc.
5. **Keep them focused**: each composable should handle one concern

---

## 6. State Management with Pinia

### What Is Pinia?

Pinia is the **official state management library for Vue** (replacing Vuex). It provides:
- Simple API with no mutations (unlike Vuex)
- Full TypeScript support
- DevTools integration
- Modular by design (no nested modules -- just multiple stores)

### Defining a Store

```javascript
// stores/counter.js
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// Composition API style (recommended)
export const useCounterStore = defineStore('counter', () => {
  // State
  const count = ref(0)
  const name = ref('Counter')

  // Getters (computed)
  const doubleCount = computed(() => count.value * 2)
  const isPositive = computed(() => count.value > 0)

  // Actions (functions)
  function increment() {
    count.value++
  }

  function decrement() {
    count.value--
  }

  function reset() {
    count.value = 0
  }

  async function incrementAsync() {
    await new Promise(resolve => setTimeout(resolve, 1000))
    count.value++
  }

  return { count, name, doubleCount, isPositive, increment, decrement, reset, incrementAsync }
})
```

```javascript
// Options API style (alternative)
export const useCounterStore = defineStore('counter', {
  state: () => ({
    count: 0,
    name: 'Counter'
  }),
  getters: {
    doubleCount: (state) => state.count * 2
  },
  actions: {
    increment() {
      this.count++
    }
  }
})
```

### A Realistic Store -- Auth

```javascript
// stores/auth.js
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'

export const useAuthStore = defineStore('auth', () => {
  const user = ref(null)
  const token = ref(localStorage.getItem('token'))

  const isAuthenticated = computed(() => !!token.value)
  const username = computed(() => user.value?.name ?? 'Guest')

  async function login(credentials) {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials)
    })

    if (!response.ok) {
      throw new Error('Login failed')
    }

    const data = await response.json()
    token.value = data.token
    user.value = data.user
    localStorage.setItem('token', data.token)
  }

  function logout() {
    token.value = null
    user.value = null
    localStorage.removeItem('token')
  }

  async function fetchUser() {
    if (!token.value) return

    const response = await fetch('/api/auth/me', {
      headers: { Authorization: `Bearer ${token.value}` }
    })

    if (response.ok) {
      user.value = await response.json()
    } else {
      logout()
    }
  }

  return { user, token, isAuthenticated, username, login, logout, fetchUser }
})
```

### Using a Store in Components

```vue
<script setup>
import { useCounterStore } from '@/stores/counter'
import { storeToRefs } from 'pinia'

const counterStore = useCounterStore()

// storeToRefs keeps reactivity when destructuring state and getters
const { count, doubleCount } = storeToRefs(counterStore)

// Actions can be destructured directly (they are plain functions)
const { increment, decrement, reset } = counterStore
</script>

<template>
  <div>
    <p>Count: {{ count }} (double: {{ doubleCount }})</p>
    <button @click="increment">+</button>
    <button @click="decrement">-</button>
    <button @click="reset">Reset</button>
  </div>
</template>
```

### Pinia vs Vuex Comparison

| Feature | Pinia | Vuex |
|---|---|---|
| Mutations | Not needed | Required (commit) |
| TypeScript | First-class | Complex types |
| Modules | Flat stores (just import) | Nested modules |
| DevTools | Full support | Full support |
| Bundle size | ~1 KB | ~6 KB |
| API | Simple, intuitive | Verbose |

---

## 7. Vue Router

### Route Definitions

```javascript
// router/index.js
import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/HomeView.vue')  // lazy-loaded
  },
  {
    path: '/about',
    name: 'About',
    component: () => import('@/views/AboutView.vue')
  },
  {
    path: '/users/:id',          // dynamic route parameter
    name: 'User',
    component: () => import('@/views/UserView.vue'),
    props: true                   // pass route params as props
  },
  {
    path: '/dashboard',
    component: () => import('@/layouts/DashboardLayout.vue'),
    meta: { requiresAuth: true },
    children: [                   // nested routes
      {
        path: '',                 // /dashboard
        name: 'DashboardHome',
        component: () => import('@/views/dashboard/HomeView.vue')
      },
      {
        path: 'settings',        // /dashboard/settings
        name: 'DashboardSettings',
        component: () => import('@/views/dashboard/SettingsView.vue')
      }
    ]
  },
  {
    path: '/:pathMatch(.*)*',    // catch-all 404
    name: 'NotFound',
    component: () => import('@/views/NotFoundView.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
```

### Navigation Guards

```javascript
// Global guard -- auth check
router.beforeEach((to, from) => {
  const authStore = useAuthStore()

  if (to.meta.requiresAuth && !authStore.isAuthenticated) {
    return { name: 'Login', query: { redirect: to.fullPath } }
  }
})

// Per-route guard
{
  path: '/admin',
  component: AdminView,
  beforeEnter: (to, from) => {
    const authStore = useAuthStore()
    if (!authStore.user?.isAdmin) {
      return { name: 'Home' }
    }
  }
}
```

### Using the Router in Components

```vue
<script setup>
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

// Read current route params
console.log(route.params.id)
console.log(route.query.search)

// Programmatic navigation
function goToUser(userId) {
  router.push({ name: 'User', params: { id: userId } })
}

function goBack() {
  router.back()
}
</script>

<template>
  <nav>
    <!-- Declarative navigation -->
    <RouterLink to="/">Home</RouterLink>
    <RouterLink :to="{ name: 'About' }">About</RouterLink>
  </nav>

  <!-- Route content renders here -->
  <RouterView />
</template>
```

---

## 8. Template Syntax & Directives

### Core Directives

```vue
<template>
  <!-- v-if / v-else-if / v-else: Conditional rendering (adds/removes from DOM) -->
  <div v-if="status === 'loading'">Loading...</div>
  <div v-else-if="status === 'error'">Error occurred</div>
  <div v-else>Content loaded</div>

  <!-- v-show: Toggle visibility via CSS display (stays in DOM) -->
  <div v-show="isVisible">Visible when isVisible is true</div>

  <!-- v-for: List rendering (always use :key) -->
  <ul>
    <li v-for="item in items" :key="item.id">
      {{ item.name }}
    </li>
  </ul>

  <!-- v-for with index -->
  <div v-for="(item, index) in items" :key="item.id">
    {{ index }}: {{ item.name }}
  </div>

  <!-- v-bind: Dynamic attribute binding (shorthand is :) -->
  <img v-bind:src="imageUrl" />
  <img :src="imageUrl" />
  <a :href="link" :class="{ active: isActive }">Link</a>

  <!-- v-on: Event handling (shorthand is @) -->
  <button v-on:click="handleClick">Click</button>
  <button @click="handleClick">Click</button>
  <input @keyup.enter="submitForm" />
  <form @submit.prevent="onSubmit">...</form>

  <!-- v-model: Two-way data binding for form elements -->
  <input v-model="name" />
  <textarea v-model="bio"></textarea>
  <select v-model="selected">
    <option value="a">A</option>
    <option value="b">B</option>
  </select>
  <input type="checkbox" v-model="agreed" />

  <!-- v-model modifiers -->
  <input v-model.trim="name" />         <!-- trims whitespace -->
  <input v-model.number="age" />         <!-- casts to number -->
  <input v-model.lazy="search" />        <!-- syncs on change, not input -->

  <!-- v-html: Render raw HTML (be cautious of XSS) -->
  <div v-html="rawHtml"></div>

  <!-- v-text: Set text content -->
  <span v-text="message"></span>
</template>
```

### v-if vs v-show

| Aspect | `v-if` | `v-show` |
|---|---|---|
| Mechanism | Adds/removes from DOM | Toggles `display: none` |
| Initial cost | Lower (if false, not rendered) | Higher (always rendered) |
| Toggle cost | Higher (destroys/creates DOM) | Lower (CSS only) |
| Use when | Condition rarely changes | Condition toggles frequently |

### Class and Style Bindings

```vue
<template>
  <!-- Object syntax for :class -->
  <div :class="{ active: isActive, 'text-bold': isBold }">...</div>

  <!-- Array syntax for :class -->
  <div :class="[baseClass, isActive ? 'active' : '']">...</div>

  <!-- Object syntax for :style -->
  <div :style="{ color: textColor, fontSize: size + 'px' }">...</div>

  <!-- Array syntax for :style -->
  <div :style="[baseStyles, overrideStyles]">...</div>
</template>
```

### Event Handling Patterns

```vue
<script setup>
import { ref } from 'vue'

const count = ref(0)

function handleClick(event) {
  console.log('Native event:', event)
}

function handleItemClick(id, event) {
  console.log('Item:', id, 'Event:', event)
}
</script>

<template>
  <!-- Inline handler -->
  <button @click="count++">Increment</button>

  <!-- Method handler -->
  <button @click="handleClick">Click Me</button>

  <!-- Passing arguments -->
  <button @click="handleItemClick(42, $event)">Item 42</button>

  <!-- Event modifiers -->
  <form @submit.prevent="onSubmit">...</form>        <!-- preventDefault -->
  <a @click.stop="doThis">...</a>                     <!-- stopPropagation -->
  <div @click.self="doThat">...</div>                  <!-- only if target is self -->
  <button @click.once="doOnce">...</button>            <!-- fires only once -->
  <input @keyup.enter="submit" />                      <!-- key modifier -->
  <input @keyup.ctrl.enter="submitWithCtrl" />         <!-- combo -->
</template>
```

---

## 9. Building an AI-Powered Vue Frontend

This section demonstrates practical patterns for an AI chat interface -- a realistic use case
for a backend AI engineer who needs to build a quick prototype frontend.

### Chat Interface Component

```vue
<!-- views/ChatView.vue -->
<script setup>
import { ref, nextTick, onMounted } from 'vue'

const messages = ref([])
const userInput = ref('')
const isStreaming = ref(false)
const chatContainer = ref(null)

async function sendMessage() {
  const text = userInput.value.trim()
  if (!text || isStreaming.value) return

  // Add user message
  messages.value.push({ role: 'user', content: text })
  userInput.value = ''
  isStreaming.value = true

  // Add placeholder for assistant response
  messages.value.push({ role: 'assistant', content: '' })
  const assistantIndex = messages.value.length - 1

  await scrollToBottom()

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: messages.value.slice(0, -1) // exclude empty placeholder
      })
    })

    // Handle streaming response (Server-Sent Events style)
    const reader = response.body.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value, { stream: true })
      messages.value[assistantIndex].content += chunk
      await scrollToBottom()
    }
  } catch (error) {
    messages.value[assistantIndex].content = `Error: ${error.message}`
  } finally {
    isStreaming.value = false
  }
}

async function scrollToBottom() {
  await nextTick()
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
}

function handleKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    sendMessage()
  }
}

onMounted(() => {
  messages.value.push({
    role: 'assistant',
    content: 'Hello! How can I help you today?'
  })
})
</script>

<template>
  <div class="chat-container">
    <div ref="chatContainer" class="messages">
      <div
        v-for="(msg, index) in messages"
        :key="index"
        :class="['message', msg.role]"
      >
        <div class="message-role">{{ msg.role === 'user' ? 'You' : 'AI' }}</div>
        <div class="message-content">{{ msg.content }}</div>
        <span
          v-if="isStreaming && index === messages.length - 1 && msg.role === 'assistant'"
          class="cursor"
        >|</span>
      </div>
    </div>

    <div class="input-area">
      <textarea
        v-model="userInput"
        @keydown="handleKeydown"
        placeholder="Type your message... (Enter to send, Shift+Enter for newline)"
        :disabled="isStreaming"
        rows="3"
      />
      <button @click="sendMessage" :disabled="isStreaming || !userInput.trim()">
        {{ isStreaming ? 'Generating...' : 'Send' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 800px;
  margin: 0 auto;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

.message {
  margin-bottom: 1rem;
  padding: 0.75rem 1rem;
  border-radius: 8px;
}

.message.user {
  background: #e3f2fd;
  margin-left: 2rem;
}

.message.assistant {
  background: #f5f5f5;
  margin-right: 2rem;
}

.message-role {
  font-weight: bold;
  font-size: 0.85rem;
  margin-bottom: 0.25rem;
}

.cursor {
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  50% { opacity: 0; }
}

.input-area {
  display: flex;
  gap: 0.5rem;
  padding: 1rem;
  border-top: 1px solid #ddd;
}

.input-area textarea {
  flex: 1;
  resize: none;
  padding: 0.5rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.input-area button {
  padding: 0.5rem 1.5rem;
  background: #1976d2;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.input-area button:disabled {
  background: #90caf9;
  cursor: not-allowed;
}
</style>
```

### File Upload for RAG

```vue
<!-- components/FileUpload.vue -->
<script setup>
import { ref } from 'vue'

const emit = defineEmits(['uploaded'])

const files = ref([])
const uploading = ref(false)
const progress = ref(0)

function handleFileSelect(event) {
  files.value = Array.from(event.target.files)
}

async function uploadFiles() {
  if (files.value.length === 0) return

  uploading.value = true
  progress.value = 0

  const formData = new FormData()
  files.value.forEach(file => {
    formData.append('files', file)
  })

  try {
    const response = await fetch('/api/rag/upload', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) throw new Error('Upload failed')

    const result = await response.json()
    emit('uploaded', result)
    files.value = []
  } catch (error) {
    console.error('Upload error:', error)
  } finally {
    uploading.value = false
  }
}
</script>

<template>
  <div class="file-upload">
    <input
      type="file"
      multiple
      accept=".pdf,.txt,.md,.csv"
      @change="handleFileSelect"
    />

    <div v-if="files.length > 0" class="file-list">
      <p>Selected files:</p>
      <ul>
        <li v-for="file in files" :key="file.name">
          {{ file.name }} ({{ (file.size / 1024).toFixed(1) }} KB)
        </li>
      </ul>
      <button @click="uploadFiles" :disabled="uploading">
        {{ uploading ? 'Uploading...' : 'Upload for RAG' }}
      </button>
    </div>
  </div>
</template>
```

### Composable for AI Chat API

```javascript
// composables/useChat.js
import { ref } from 'vue'

export function useChat(apiUrl = '/api/chat') {
  const messages = ref([])
  const isStreaming = ref(false)
  const error = ref(null)

  async function send(userMessage) {
    if (isStreaming.value) return

    error.value = null
    messages.value.push({ role: 'user', content: userMessage })
    messages.value.push({ role: 'assistant', content: '' })

    const assistantIdx = messages.value.length - 1
    isStreaming.value = true

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messages.value.slice(0, -1)
        })
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        messages.value[assistantIdx].content += decoder.decode(value, { stream: true })
      }
    } catch (err) {
      error.value = err.message
      messages.value[assistantIdx].content = `Error: ${err.message}`
    } finally {
      isStreaming.value = false
    }
  }

  function clearHistory() {
    messages.value = []
  }

  return { messages, isStreaming, error, send, clearHistory }
}
```

---

## 10. Vite & Build Tools

### Vite Basics

Vite is the default build tool for Vue 3 projects. It provides:
- **Instant dev server start** (uses native ES modules, no bundling in dev)
- **Hot Module Replacement (HMR)** that stays fast regardless of project size
- **Optimized production builds** using Rollup under the hood

```bash
# Create a new Vue project
npm create vue@latest my-project
cd my-project
npm install
npm run dev      # start dev server
npm run build    # production build
npm run preview  # preview production build locally
```

### Project Structure

```
my-project/
  index.html           # entry HTML (Vite uses this as entry point)
  vite.config.js       # Vite configuration
  src/
    main.js            # app entry point
    App.vue            # root component
    components/        # reusable components
    views/             # page-level components
    composables/       # composable functions
    stores/            # Pinia stores
    router/            # Vue Router config
    assets/            # static assets (images, fonts)
  public/              # files served as-is (favicon, robots.txt)
```

### Vite Configuration

```javascript
// vite.config.js
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],

  resolve: {
    alias: {
      '@': '/src'   // enables @/components/Foo.vue imports
    }
  },

  server: {
    port: 3000,

    // Proxy API calls to backend during development
    proxy: {
      '/api': {
        target: 'http://localhost:8000',   // FastAPI / Express backend
        changeOrigin: true
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true
      }
    }
  },

  build: {
    outDir: 'dist',
    sourcemap: true,

    rollupOptions: {
      output: {
        // Split vendor code into separate chunk for better caching
        manualChunks: {
          vue: ['vue', 'vue-router', 'pinia']
        }
      }
    }
  }
})
```

### Environment Variables

```bash
# .env                   -- loaded in all cases
VITE_APP_TITLE=My App

# .env.development       -- only in dev mode
VITE_API_URL=http://localhost:8000

# .env.production        -- only in production build
VITE_API_URL=https://api.example.com
```

```javascript
// Access in code (must be prefixed with VITE_)
console.log(import.meta.env.VITE_API_URL)
console.log(import.meta.env.VITE_APP_TITLE)
console.log(import.meta.env.MODE)   // 'development' or 'production'
console.log(import.meta.env.DEV)    // true in dev
console.log(import.meta.env.PROD)   // true in production
```

### Build Optimization Tips

1. **Lazy-load routes**: Use dynamic `import()` in router definitions
2. **Tree-shaking**: Import only what you need from libraries (`import { ref } from 'vue'`)
3. **Code splitting**: `manualChunks` in Vite config to separate vendor code
4. **Async components**: Use `defineAsyncComponent` for heavy components

```javascript
import { defineAsyncComponent } from 'vue'

const HeavyChart = defineAsyncComponent(() =>
  import('./components/HeavyChart.vue')
)
```

5. **Image optimization**: Use `vite-plugin-imagemin` or serve images from a CDN
6. **Analyze bundle**: Use `rollup-plugin-visualizer` to find large dependencies

---

## 11. Q&A Section

### Q1: What is the Composition API and how does it differ from the Options API?

**Answer**: The Composition API is a set of functions (`ref`, `reactive`, `computed`, `watch`,
`onMounted`, etc.) that allow you to organize component logic by **logical concern** rather
than by option type.

In the **Options API**, a component's logic is split across `data()`, `computed`, `methods`,
`watch`, and lifecycle hooks. If a component handles multiple features (e.g., search + pagination
+ filtering), the code for each feature is scattered across these options.

In the **Composition API**, you write all related logic together and can extract it into
**composables** (reusable functions). This results in better code organization, reusability,
and TypeScript support.

Key differences:
- Options API uses `this` to access state; Composition API uses explicit variables
- Composition API enables composables (like React hooks); Options API uses mixins (which have
  naming collisions and unclear data sources)
- Composition API has better TypeScript inference since everything is just functions and variables

---

### Q2: Explain `ref()` vs `reactive()`. When would you use each?

**Answer**:

**`ref()`** wraps any value (primitive or object) in a `{ value: ... }` container. You access
the value via `.value` in JavaScript (auto-unwrapped in templates).

**`reactive()`** returns a Proxy of the original object. It only works with objects (not
primitives). You access properties directly without `.value`.

```javascript
const count = ref(0)       // count.value to access
const state = reactive({   // state.count to access
  count: 0
})
```

**When to use each**:
- **Prefer `ref()` in most cases**: it works with any type, is explicit about reactivity
  (`.value`), and avoids the destructuring gotcha of `reactive()`
- **Use `reactive()`** when you have a group of related state properties and want to avoid
  writing `.value` everywhere -- but be aware that destructuring breaks reactivity

**Practical rule**: If you are unsure, use `ref()`. The Vue team and ecosystem lean toward `ref()`.

---

### Q3: What are composables and how do they differ from mixins?

**Answer**: Composables are functions that encapsulate reactive stateful logic using the
Composition API. They are the recommended replacement for Vue 2 mixins.

| Aspect | Composables | Mixins |
|---|---|---|
| Naming conflicts | None (explicit returns) | Possible (implicit merge) |
| Data source clarity | Obvious (you call the function) | Unclear (magic merge) |
| TypeScript | Excellent | Poor |
| Parameterization | Easy (function arguments) | Difficult |
| Reuse granularity | Fine-grained | All-or-nothing |

---

### Q4: How does Vue 3 reactivity work under the hood?

**Answer**: Vue 3 uses **ES6 Proxy** objects to intercept reads (GET traps) and writes (SET traps)
on reactive data.

1. When a component renders and accesses reactive data, the Proxy's GET trap fires.
   Vue records a dependency: "this component depends on this property" (**track**).
2. When the data is mutated, the Proxy's SET trap fires. Vue finds all components that depend
   on that property and schedules them for re-render (**trigger**).
3. Re-renders are batched and asynchronous (via `nextTick`) for performance.

Advantages over Vue 2's `Object.defineProperty`:
- Detects new property additions and deletions
- Detects array index mutations and `.length` changes
- Works on Maps, Sets, WeakMaps, WeakSets
- Lazy: only creates proxies for nested objects when they are accessed

---

### Q5: What is Pinia and how does it compare to Vuex?

**Answer**: Pinia is the **official state management library for Vue 3** (maintained by the Vue
core team). It replaces Vuex.

Key improvements over Vuex:
- **No mutations**: Just use actions to modify state directly (simpler mental model)
- **Flat architecture**: No nested modules; each store is independent
- **TypeScript-first**: Excellent type inference without extra type declarations
- **Composition API support**: Stores can use `ref`, `computed`, etc.
- **Smaller bundle**: ~1 KB vs ~6 KB for Vuex
- **DevTools integration**: Full support for time-travel debugging

```javascript
// Pinia store is just a function
export const useUserStore = defineStore('user', () => {
  const name = ref('Alice')
  const upperName = computed(() => name.value.toUpperCase())
  function setName(newName) { name.value = newName }
  return { name, upperName, setName }
})
```

---

### Q6: How do you handle component communication in Vue?

**Answer**: There are several patterns, each suited to different scenarios:

1. **Props (parent -> child)**: Pass data down through props. One-way data flow.
2. **Emits (child -> parent)**: Child fires events with `emit()`, parent listens with `@event`.
3. **v-model (two-way)**: Syntactic sugar for prop + emit. Used for form-like components.
4. **Provide/Inject (ancestor -> descendant)**: Share data across many levels without prop
   drilling. Good for themes, auth state, configuration.
5. **Pinia (any -> any)**: For truly global state shared across unrelated components.
6. **Template refs**: Direct access to child component instances (use sparingly).

**Rule of thumb**: Start with props/emits. Use provide/inject for deep nesting. Use Pinia for
state shared across multiple unrelated components.

---

### Q7: What are slots and when should you use them?

**Answer**: Slots allow a parent component to inject content into a child component's template.
They enable flexible, reusable components.

- **Default slot**: A single content injection point (like React's `children`)
- **Named slots**: Multiple injection points identified by name
- **Scoped slots**: The child exposes data to the parent for rendering

**When to use**:
- Building UI library components (Card, Modal, Layout)
- Creating flexible wrappers where the parent controls what's rendered inside
- Building "renderless" components that provide logic but let the parent control presentation

---

### Q8: How do you handle async operations in Vue?

**Answer**: Several patterns:

1. **In lifecycle hooks**: Use `onMounted` with `async/await`
   ```javascript
   onMounted(async () => {
     const data = await fetchData()
   })
   ```

2. **In actions**: Pinia actions can be async
   ```javascript
   async function fetchUsers() {
     loading.value = true
     try { users.value = await api.getUsers() }
     finally { loading.value = false }
   }
   ```

3. **In composables**: Create reusable async logic (like `useFetch`)

4. **`<Suspense>`** (experimental): Wraps components with async `setup()` and shows
   fallback content while loading
   ```vue
   <Suspense>
     <template #default><AsyncComponent /></template>
     <template #fallback><LoadingSpinner /></template>
   </Suspense>
   ```

---

### Q9: What are lifecycle hooks in the Composition API?

**Answer**: Lifecycle hooks let you run code at specific points in a component's life.

- `onBeforeMount` / `onMounted` -- before/after the component is added to the DOM
- `onBeforeUpdate` / `onUpdated` -- before/after the component re-renders
- `onBeforeUnmount` / `onUnmounted` -- before/after the component is removed from the DOM
- `onErrorCaptured` -- when an error from a descendant is caught

The `setup()` function itself replaces `beforeCreate` and `created`. Code in `<script setup>`
runs during these phases.

Common usage: `onMounted` for DOM access and initial data fetching; `onUnmounted` for cleanup
(removing event listeners, cancelling timers, closing connections).

---

### Q10: How do you optimize Vue app performance?

**Answer**:

1. **Lazy-load routes**: Dynamic `import()` so each page is a separate chunk
2. **`v-once`**: Render a subtree only once (static content)
3. **`v-memo`**: Cache a template subtree based on dependency array
4. **`shallowRef` / `shallowReactive`**: Skip deep reactivity for large data
5. **Virtual scrolling**: Use libraries like `vue-virtual-scroller` for long lists
6. **`computed` over methods**: Computed values are cached; method calls in templates run every render
7. **`defineAsyncComponent`**: Load heavy components on demand
8. **Keep `v-if` vs `v-show` in mind**: Use `v-show` for frequently toggled elements
9. **Avoid unnecessary watchers**: Use `computed` when possible
10. **Key attribute on `v-for`**: Helps Vue reuse existing DOM elements efficiently
11. **Tree-shaking**: Import only what you need (Vue 3 is fully tree-shakable)

---

### Q11: What is `nextTick()` and when do you need it?

**Answer**: `nextTick()` returns a promise that resolves after the next DOM update cycle. Vue
batches reactive state changes and updates the DOM asynchronously. If you need to access the
updated DOM immediately after changing state, use `nextTick()`.

```javascript
import { ref, nextTick } from 'vue'

const message = ref('Hello')
message.value = 'Updated'

// DOM has NOT been updated yet here
await nextTick()
// DOM is now updated
```

**Common use case**: Scrolling to the bottom of a chat container after adding a new message.

---

### Q12: How does `v-model` work on custom components?

**Answer**: On native elements, `v-model` is syntactic sugar for `:value` + `@input`. On custom
components, Vue 3 uses `:modelValue` prop + `@update:modelValue` event.

```vue
<!-- Parent: these are equivalent -->
<CustomInput v-model="text" />
<CustomInput :modelValue="text" @update:modelValue="text = $event" />
```

You can have multiple v-model bindings with names:
```vue
<UserForm v-model:firstName="first" v-model:lastName="last" />
```

Each named v-model maps to `:firstName` prop + `@update:firstName` event (and similarly for
`lastName`).

---

### Q13: What is `<Teleport>` and when would you use it?

**Answer**: `<Teleport>` renders its children in a different location in the DOM tree than
where the component exists in the component tree. This is useful for modals, tooltips,
and notifications that need to be rendered outside of a parent with `overflow: hidden` or
a specific z-index context.

```vue
<template>
  <button @click="showModal = true">Open Modal</button>

  <Teleport to="body">
    <div v-if="showModal" class="modal-overlay">
      <div class="modal">
        <p>This is rendered at the end of body</p>
        <button @click="showModal = false">Close</button>
      </div>
    </div>
  </Teleport>
</template>
```

The component logic (state, events) stays in the component tree; only the DOM rendering
is moved.

---

### Q14: What is the difference between `v-if` and `v-show`?

**Answer**:
- **`v-if`**: Conditionally renders the element. When false, the element is completely removed
  from the DOM. Has a higher toggle cost (destroying and recreating DOM).
- **`v-show`**: Always renders the element but toggles `display: none`. Has a higher initial
  cost (always rendered) but a lower toggle cost.

**Use `v-if`** when the condition rarely changes or when the hidden content is expensive to
render (it avoids the initial cost).

**Use `v-show`** when you toggle visibility frequently (like a dropdown menu).

---

### Q15: How do you handle form validation in Vue?

**Answer**: There are several approaches:

1. **Manual validation** using computed properties and watchers:
   ```javascript
   const email = ref('')
   const emailError = computed(() => {
     if (!email.value) return 'Required'
     if (!email.value.includes('@')) return 'Invalid email'
     return ''
   })
   const isValid = computed(() => !emailError.value)
   ```

2. **Libraries**: VeeValidate (most popular, works with Composition API and Zod/Yup schemas),
   FormKit (opinionated but powerful).

3. **Native HTML validation**: Use `required`, `minlength`, `pattern` attributes on form
   elements combined with Vue's `@submit.prevent`.

---

### Q16: What is `defineExpose()` and when do you need it?

**Answer**: With `<script setup>`, all bindings are private by default -- they are not accessible
from parent components via template refs. `defineExpose()` explicitly exposes selected bindings.

```vue
<!-- ChildComponent.vue -->
<script setup>
import { ref } from 'vue'

const internalState = ref('private')
const publicMethod = () => console.log('called from parent')

defineExpose({ publicMethod })
</script>
```

```vue
<!-- Parent.vue -->
<script setup>
import { ref, onMounted } from 'vue'
import ChildComponent from './ChildComponent.vue'

const childRef = ref(null)

onMounted(() => {
  childRef.value.publicMethod() // works
  // childRef.value.internalState // undefined -- not exposed
})
</script>

<template>
  <ChildComponent ref="childRef" />
</template>
```

---

### Q17: How do you handle error boundaries in Vue?

**Answer**: Vue 3 provides `onErrorCaptured` lifecycle hook and the `errorHandler` global config.

```javascript
// Global error handler (main.js)
app.config.errorHandler = (err, instance, info) => {
  console.error('Global error:', err)
  // Send to error tracking service
}
```

```vue
<!-- ErrorBoundary.vue -->
<script setup>
import { ref, onErrorCaptured } from 'vue'

const error = ref(null)

onErrorCaptured((err, instance, info) => {
  error.value = err
  return false // stop propagation
})
</script>

<template>
  <div v-if="error" class="error">
    Something went wrong: {{ error.message }}
  </div>
  <slot v-else />
</template>
```

---

### Q18: What is the purpose of `key` in `v-for`?

**Answer**: The `key` attribute helps Vue's virtual DOM algorithm identify which elements have
changed, been added, or been removed. Without `key`, Vue uses a "minimum mutation" strategy
that may reuse DOM elements incorrectly (leading to stale state in child components).

**Rules**:
- Always provide a unique, stable `key` for `v-for` items
- Use a unique identifier (like `item.id`), never the array index (indices shift on insert/delete)
- `key` can also be used on components to force re-creation instead of reuse

```vue
<!-- Good -->
<li v-for="user in users" :key="user.id">{{ user.name }}</li>

<!-- Bad: index can cause issues on reorder/insert/delete -->
<li v-for="(user, index) in users" :key="index">{{ user.name }}</li>
```

---

### Q19: How do you test Vue components?

**Answer**: The standard testing stack for Vue 3:

- **Vitest**: Unit test runner (Vite-native, fast, Jest-compatible API)
- **Vue Test Utils**: Official testing utility library
- **Testing Library (Vue)**: User-centric testing (queries by text, role, label)

```javascript
// Example with Vitest + Vue Test Utils
import { mount } from '@vue/test-utils'
import { describe, it, expect } from 'vitest'
import Counter from './Counter.vue'

describe('Counter', () => {
  it('increments count on click', async () => {
    const wrapper = mount(Counter)

    expect(wrapper.text()).toContain('Count: 0')

    await wrapper.find('button').trigger('click')

    expect(wrapper.text()).toContain('Count: 1')
  })

  it('accepts initial count prop', () => {
    const wrapper = mount(Counter, {
      props: { initialCount: 10 }
    })

    expect(wrapper.text()).toContain('Count: 10')
  })
})
```

---

### Q20: As a backend AI engineer, when and why would you choose Vue.js for a frontend?

**Answer**: Vue.js is a strong choice for backend engineers building frontends because:

1. **Low learning curve**: Vue's template syntax is close to plain HTML. The Composition API
   is function-based and feels natural to Python/backend developers.

2. **Single File Components**: Everything in one file (.vue) -- no need to juggle multiple
   files per component.

3. **Batteries included**: Vue Router, Pinia, and Vite provide a complete stack with official
   support and minimal decision fatigue.

4. **AI/ML prototyping**: For building chat interfaces, dashboards, or RAG frontends, Vue's
   reactive system and composables make it easy to handle streaming responses, real-time
   updates, and complex state.

5. **Gradual adoption**: You can start with a simple script tag and scale to a full SPA
   as the project grows.

**Practical scenario**: You have a FastAPI backend serving an LLM-powered API. You need a chat
interface with streaming support, file upload for RAG, and a simple dashboard. Vue 3 + Vite
gives you a fast dev setup, reactive state management for streaming tokens, and composables
to encapsulate the chat/upload logic cleanly.

---

## Quick Reference Card

```
Vue 3 Composition API Cheatsheet
---------------------------------
State:          ref(value), reactive(obj)
Computed:       computed(() => ...)
Watch:          watch(source, callback), watchEffect(callback)
Lifecycle:      onMounted, onUpdated, onUnmounted, onBeforeMount, ...
Props:          defineProps({ name: Type })
Emits:          defineEmits(['event'])
Expose:         defineExpose({ method })
Slots:          <slot>, <slot name="x">, <slot :data="x">
Provide:        provide('key', value)
Inject:         inject('key', defaultValue)
Router:         useRouter(), useRoute()
Store:          defineStore('id', () => { ... })
Next tick:      await nextTick()
Template ref:   const el = ref(null) + <div ref="el">
Async comp:     defineAsyncComponent(() => import('./X.vue'))
Teleport:       <Teleport to="body">...</Teleport>
```
