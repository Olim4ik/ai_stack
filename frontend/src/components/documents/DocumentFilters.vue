<script setup lang="ts">
const emit = defineEmits<{
  filter: [team: string]
}>()

defineProps<{
  selectedTeam: string
}>()

const teams = ['', 'platform', 'backend', 'frontend', 'devops', 'data']
</script>

<template>
  <div class="flex items-center gap-3">
    <label class="text-sm font-medium text-gray-600">Filter by team:</label>
    <select
      :value="selectedTeam"
      @change="emit('filter', ($event.target as HTMLSelectElement).value)"
      class="rounded-md border border-gray-300 px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
    >
      <option v-for="team in teams" :key="team" :value="team">
        {{ team || 'All teams' }}
      </option>
    </select>
  </div>
</template>
