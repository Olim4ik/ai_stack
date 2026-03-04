# 02. SQL: Indexes, Joins, Queries, Sharding & ACID

> Comprehensive interview preparation for a Backend / AI Engineer role.
> Covers theory, visual diagrams, practical SQL, Python integration, and 25+ Q&A.

---

## Table of Contents

1. [SQL Fundamentals](#1-sql-fundamentals)
2. [Joins](#2-joins)
3. [Indexes](#3-indexes)
4. [ACID Properties](#4-acid-properties)
5. [Sharding](#5-sharding)
6. [Database Design](#6-database-design)
7. [Performance Optimization](#7-performance-optimization)
8. [PostgreSQL Specifics](#8-postgresql-specifics)
9. [SQL in Python](#9-sql-in-python)
10. [Q&A Section (25 Questions)](#10-qa-section)

---

## 1. SQL Fundamentals

### 1.1 Core DML Statements

```sql
-- ============================================================
-- SELECT — retrieve data
-- ============================================================
SELECT id, name, email
FROM   users
WHERE  created_at >= '2025-01-01'
ORDER  BY name ASC
LIMIT  20 OFFSET 40;           -- page 3, 20 rows per page

-- ============================================================
-- INSERT — add rows
-- ============================================================
INSERT INTO users (name, email, role)
VALUES ('Alice', 'alice@example.com', 'engineer');

-- Insert multiple rows in one statement (much faster than N inserts)
INSERT INTO users (name, email, role)
VALUES
    ('Bob',   'bob@example.com',   'manager'),
    ('Carol', 'carol@example.com', 'engineer');

-- Insert from a subquery
INSERT INTO user_archive (id, name, email)
SELECT id, name, email
FROM   users
WHERE  deleted_at IS NOT NULL;

-- ============================================================
-- UPDATE — modify existing rows
-- ============================================================
UPDATE users
SET    role = 'senior_engineer',
       updated_at = NOW()
WHERE  id = 42;

-- Update with a join (PostgreSQL syntax)
UPDATE orders o
SET    status = 'cancelled'
FROM   users u
WHERE  o.user_id = u.id
  AND  u.is_banned = TRUE;

-- ============================================================
-- DELETE — remove rows
-- ============================================================
DELETE FROM users
WHERE  id = 42;

-- Delete with a subquery
DELETE FROM orders
WHERE  user_id IN (
    SELECT id FROM users WHERE is_banned = TRUE
);
```

### 1.2 Filtering & Aggregation

```
Execution order of a SELECT statement:

  1. FROM / JOIN        — assemble the source tables
  2. WHERE              — filter individual rows
  3. GROUP BY           — collapse rows into groups
  4. HAVING             — filter groups
  5. SELECT             — evaluate expressions / aliases
  6. DISTINCT           — remove duplicate rows
  7. ORDER BY           — sort the result set
  8. LIMIT / OFFSET     — paginate
```

```sql
-- GROUP BY + HAVING
SELECT   department,
         COUNT(*)       AS employee_count,
         AVG(salary)    AS avg_salary
FROM     employees
GROUP BY department
HAVING   COUNT(*) > 5           -- filter AFTER grouping
ORDER BY avg_salary DESC;

-- WHERE vs HAVING:
--   WHERE  filters rows BEFORE grouping  (cannot reference aggregates)
--   HAVING filters groups AFTER grouping  (can reference aggregates)
```

### 1.3 Subqueries & CTEs

```sql
-- Correlated subquery — runs once per outer row (can be slow)
SELECT e.name, e.salary
FROM   employees e
WHERE  e.salary > (
    SELECT AVG(salary)
    FROM   employees
    WHERE  department = e.department   -- correlated: references outer row
);

-- CTE (Common Table Expression) — WITH clause
-- Easier to read, can be referenced multiple times
WITH dept_avg AS (
    SELECT department, AVG(salary) AS avg_sal
    FROM   employees
    GROUP  BY department
)
SELECT e.name, e.salary, d.avg_sal
FROM   employees e
JOIN   dept_avg  d ON e.department = d.department
WHERE  e.salary > d.avg_sal;

-- Recursive CTE — traversing a tree / hierarchy
WITH RECURSIVE org_tree AS (
    -- Base case: the CEO (no manager)
    SELECT id, name, manager_id, 1 AS depth
    FROM   employees
    WHERE  manager_id IS NULL

    UNION ALL

    -- Recursive case: employees who report to someone already in the tree
    SELECT e.id, e.name, e.manager_id, t.depth + 1
    FROM   employees e
    JOIN   org_tree  t ON e.manager_id = t.id
)
SELECT * FROM org_tree ORDER BY depth, name;
```

### 1.4 Window Functions

Window functions compute a value for every row based on a "window" of related
rows, **without collapsing** the result set (unlike GROUP BY).

```
Syntax:
  function_name(...) OVER (
      [PARTITION BY col1, col2, ...]   -- define groups (windows)
      [ORDER BY col3, ...]             -- order within each window
      [ROWS BETWEEN ... AND ...]       -- optional frame clause
  )
```

```sql
-- ROW_NUMBER / RANK / DENSE_RANK
SELECT
    name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS row_num,
    RANK()       OVER (PARTITION BY department ORDER BY salary DESC) AS rnk,
    DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dense_rnk
FROM employees;

-- Example output (Engineering department):
-- ┌──────────┬─────────────┬────────┬─────────┬─────┬───────────┐
-- │ name     │ department  │ salary │ row_num │ rnk │ dense_rnk │
-- ├──────────┼─────────────┼────────┼─────────┼─────┼───────────┤
-- │ Alice    │ Engineering │ 150000 │       1 │   1 │         1 │
-- │ Bob      │ Engineering │ 150000 │       2 │   1 │         1 │  ← tie
-- │ Carol    │ Engineering │ 130000 │       3 │   3 │         2 │
-- │ Dave     │ Engineering │ 120000 │       4 │   4 │         3 │
-- └──────────┴─────────────┴────────┴─────────┴─────┴───────────┘
--
-- ROW_NUMBER: always unique (1, 2, 3, 4)
-- RANK:       ties get the same rank, then skip (1, 1, 3, 4)
-- DENSE_RANK: ties get the same rank, no skip  (1, 1, 2, 3)
```

```sql
-- LAG / LEAD — access previous or next row
SELECT
    date,
    revenue,
    LAG(revenue, 1)  OVER (ORDER BY date) AS prev_day_revenue,
    LEAD(revenue, 1) OVER (ORDER BY date) AS next_day_revenue,
    revenue - LAG(revenue, 1) OVER (ORDER BY date) AS day_over_day_change
FROM daily_sales;

-- Running total with SUM window
SELECT
    date,
    revenue,
    SUM(revenue) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        AS running_total
FROM daily_sales;

-- Top-N per group (common pattern)
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rn
    FROM employees
)
SELECT * FROM ranked WHERE rn <= 3;   -- top 3 earners per department
```

### 1.5 CASE Expressions

```sql
SELECT
    name,
    salary,
    CASE
        WHEN salary >= 150000 THEN 'Senior'
        WHEN salary >= 100000 THEN 'Mid'
        WHEN salary >=  60000 THEN 'Junior'
        ELSE 'Intern'
    END AS level,
    -- CASE in aggregation: pivot-like behavior
    COUNT(*) FILTER (WHERE status = 'active')  AS active_count   -- PostgreSQL
FROM employees
GROUP BY department;

-- Conditional aggregation (portable across databases)
SELECT
    department,
    SUM(CASE WHEN gender = 'M' THEN 1 ELSE 0 END) AS male_count,
    SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) AS female_count
FROM employees
GROUP BY department;
```

---

## 2. Joins

### 2.1 Visual Overview

```
Sample data:

Table A (users)       Table B (orders)
┌────┬───────┐       ┌────┬─────────┐
│ id │ name  │       │ id │ user_id │
├────┼───────┤       ├────┼─────────┤
│  1 │ Alice │       │ 10 │    2    │
│  2 │ Bob   │       │ 11 │    3    │
│  3 │ Carol │       │ 12 │    4    │  ← user_id=4 has no matching user
└────┴───────┘       └────┴─────────┘

Simplified set view (matching user_ids):
  A has: {1, 2, 3}
  B has: {2, 3, 4}
```

```
┌──────────────────────────────────────────────────────────────────────┐
│                          JOIN TYPES                                  │
├─────────────────┬────────────────────────────────────────────────────┤
│                 │                                                    │
│  INNER JOIN     │   ╔═══╗                                           │
│  {2, 3}         │  ┌┤   ├┐    Only rows that match in BOTH tables   │
│                 │  │╚═══╝│                                           │
│                 │  A     B                                           │
│                 │                                                    │
├─────────────────┼────────────────────────────────────────────────────┤
│                 │                                                    │
│  LEFT JOIN      │  ╔════╗                                            │
│  {1, 2, 3}      │  ┤    ├┐    ALL rows from A, matched rows from B  │
│                 │  ╚════╝│    NULLs where B has no match             │
│                 │  A     B                                           │
│                 │                                                    │
├─────────────────┼────────────────────────────────────────────────────┤
│                 │                                                    │
│  RIGHT JOIN     │       ╔════╗                                       │
│  {2, 3, 4}      │  ┌    ┤    ├  ALL rows from B, matched from A     │
│                 │  │    ╚════╝  NULLs where A has no match           │
│                 │  A     B                                           │
│                 │                                                    │
├─────────────────┼────────────────────────────────────────────────────┤
│                 │                                                    │
│  FULL OUTER     │  ╔═════════╗                                       │
│  {1, 2, 3, 4}   │  ┤         ├  ALL rows from BOTH tables           │
│                 │  ╚═════════╝  NULLs on both sides where no match  │
│                 │  A        B                                        │
│                 │                                                    │
├─────────────────┼────────────────────────────────────────────────────┤
│                 │                                                    │
│  CROSS JOIN     │  Every row of A combined with every row of B       │
│  3 x 3 = 9     │  No ON clause. Produces |A| * |B| rows.           │
│                 │                                                    │
└─────────────────┴────────────────────────────────────────────────────┘
```

### 2.2 SQL Examples for Each Join

```sql
-- INNER JOIN — only matching rows
SELECT u.name, o.id AS order_id
FROM   users  u
INNER JOIN orders o ON u.id = o.user_id;
-- Result:
-- Bob   | 10
-- Carol | 11

-- LEFT JOIN — all users, even those without orders
SELECT u.name, o.id AS order_id
FROM   users  u
LEFT JOIN orders o ON u.id = o.user_id;
-- Result:
-- Alice | NULL    ← no matching order
-- Bob   | 10
-- Carol | 11

-- RIGHT JOIN — all orders, even those without a matching user
SELECT u.name, o.id AS order_id
FROM   users  u
RIGHT JOIN orders o ON u.id = o.user_id;
-- Result:
-- Bob   | 10
-- Carol | 11
-- NULL  | 12      ← user_id=4 does not exist in users

-- FULL OUTER JOIN — everything from both sides
SELECT u.name, o.id AS order_id
FROM   users  u
FULL OUTER JOIN orders o ON u.id = o.user_id;
-- Result:
-- Alice | NULL
-- Bob   | 10
-- Carol | 11
-- NULL  | 12

-- CROSS JOIN — cartesian product (3 users x 3 orders = 9 rows)
SELECT u.name, o.id AS order_id
FROM   users u
CROSS JOIN orders o;
```

### 2.3 Self Join

A self join joins a table to itself. Common for hierarchical data.

```sql
-- Find employees and their managers
SELECT
    e.name  AS employee,
    m.name  AS manager
FROM   employees e
LEFT JOIN employees m ON e.manager_id = m.id;

-- Result:
-- ┌──────────┬─────────┐
-- │ employee │ manager │
-- ├──────────┼─────────┤
-- │ Alice    │ NULL    │  ← CEO, no manager
-- │ Bob      │ Alice   │
-- │ Carol    │ Alice   │
-- │ Dave     │ Bob     │
-- └──────────┴─────────┘
```

### 2.4 Anti-Joins (LEFT JOIN WHERE NULL)

An anti-join finds rows in A that have **no match** in B.

```sql
-- Find users who have NEVER placed an order
-- Method 1: LEFT JOIN + WHERE NULL  (anti-join pattern)
SELECT u.*
FROM   users  u
LEFT JOIN orders o ON u.id = o.user_id
WHERE  o.id IS NULL;

-- Method 2: NOT EXISTS (often same performance)
SELECT u.*
FROM   users u
WHERE  NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.id
);

-- Method 3: NOT IN (be careful with NULLs!)
SELECT u.*
FROM   users u
WHERE  u.id NOT IN (
    SELECT user_id FROM orders WHERE user_id IS NOT NULL
);
-- WARNING: If the subquery returns any NULL, NOT IN returns
-- no rows at all. Always add WHERE col IS NOT NULL.
```

### 2.5 Semi-Join

A semi-join finds rows in A that **have at least one match** in B, but does
not duplicate rows if there are multiple matches.

```sql
-- Find users who have placed at least one order
SELECT u.*
FROM   users u
WHERE  EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.id
);
-- This is preferred over INNER JOIN + DISTINCT when you don't need order data.
```

---

## 3. Indexes

### 3.1 How a B-Tree Index Works

A B-tree (balanced tree) keeps data sorted and allows search, insert, and
delete in **O(log n)** time. Every leaf node is at the same depth.

```
B-Tree index on column "age":

                        ┌───────────┐
              Level 0   │    [40]    │            root
                        └─────┬─────┘
                    ┌─────────┴─────────┐
                    v                   v
              ┌──────────┐       ┌──────────┐
   Level 1    │ [10, 25] │       │ [55, 70] │    internal nodes
              └────┬─────┘       └────┬─────┘
           ┌───┬───┘  └──┐     ┌───┬─┘  └──┐
           v   v      v  v     v   v     v  v
         ┌───┬───┬─────┬───┬─────┬───┬─────┬───┐
Level 2  │ 3 │10 │ 15  │25 │ 30  │55 │ 60  │70 │   leaf nodes
         │ 7 │12 │ 20  │28 │ 35  │58 │ 65  │80 │   (sorted, doubly linked)
         └───┴───┴─────┴───┴─────┴───┴─────┴───┘
           ^───────>──────>──────>──────>───────^
                leaf nodes linked for range scans

Search for age = 28:
  root [40]  → 28 < 40 → go left
  [10, 25]   → 28 > 25 → go right-most child
  leaf [25, 28] → found!   (3 page reads for millions of rows)
```

**Why B-tree and not binary tree?** Each node holds many keys (fan-out of
100-500), so the tree is very shallow. A table with 100 million rows typically
needs only 3-4 levels, meaning 3-4 disk I/Os per lookup.

### 3.2 Index Types

| Type | Best For | Data Structure | Example Use |
|------|----------|----------------|-------------|
| **B-tree** | Equality, range, sorting, prefix LIKE | Balanced tree | `WHERE age > 25`, `ORDER BY name` |
| **Hash** | Equality only (=) | Hash table | `WHERE id = 42` |
| **GIN** | Full-text search, JSONB, arrays | Inverted index | `WHERE tags @> '{python}'` |
| **GiST** | Geometric, range, nearest-neighbor | Generalized search tree | PostGIS, `WHERE point <-> target` |
| **BRIN** | Very large tables with natural ordering | Block range summaries | Time-series `WHERE ts > '2025-01-01'` |

```sql
-- Create each type
CREATE INDEX idx_users_email     ON users USING btree (email);
CREATE INDEX idx_users_id_hash   ON users USING hash  (id);
CREATE INDEX idx_docs_content    ON docs  USING gin   (to_tsvector('english', content));
CREATE INDEX idx_geo_location    ON places USING gist (location);
CREATE INDEX idx_events_ts       ON events USING brin (created_at);
```

### 3.3 Composite Indexes & Column Order

A composite index indexes multiple columns. **Column order matters enormously.**

```sql
CREATE INDEX idx_orders_user_status ON orders (user_id, status);
```

```
Composite B-tree (user_id, status):

              ┌──────────────────────────────┐
              │        (user_id=5)           │
              └──────────────┬───────────────┘
         ┌───────────────────┴───────────────────┐
         v                                       v
  ┌──────────────────┐                 ┌──────────────────┐
  │ user_id < 5      │                 │ user_id >= 5     │
  │ sorted by status │                 │ sorted by status │
  └──────────────────┘                 └──────────────────┘

Think of it as a phone book sorted by (last_name, first_name).
You can quickly find:
  - All Smiths                     → WHERE user_id = 5
  - Smith, Alice                   → WHERE user_id = 5 AND status = 'active'
  - All entries after Smith        → WHERE user_id > 5

You CANNOT efficiently find:
  - All Alices (any last name)     → WHERE status = 'active'  (index not useful)
```

**The "leftmost prefix" rule**: A composite index `(A, B, C)` can serve queries on:
- `(A)` alone
- `(A, B)` together
- `(A, B, C)` together
- But NOT `(B)` alone, `(C)` alone, or `(B, C)` alone.

```sql
-- These USE the index (A, B, C):
WHERE A = 1
WHERE A = 1 AND B = 2
WHERE A = 1 AND B = 2 AND C = 3
WHERE A = 1 AND B > 5               -- range on B, uses A and B

-- These DO NOT use the index (or only partially):
WHERE B = 2                          -- skips A
WHERE C = 3                          -- skips A and B
WHERE B = 2 AND C = 3               -- skips A
```

### 3.4 Covering Indexes (Index-Only Scans)

A covering index includes all columns that a query needs, so the database
never has to visit the actual table ("heap"). This is called an **index-only scan**.

```sql
-- Query: find user emails by name
SELECT email FROM users WHERE name = 'Alice';

-- Regular index: look up name in index → get row pointer → fetch row from heap → read email
-- Covering index: look up name in index → email is already IN the index → done!

CREATE INDEX idx_users_name_email ON users (name) INCLUDE (email);
-- The INCLUDE clause adds "email" to the leaf nodes without affecting sort order.

-- PostgreSQL EXPLAIN will show:
--   Index Only Scan using idx_users_name_email
```

### 3.5 When Indexes Help vs Hurt

```
INDEXES HELP when:                    INDEXES HURT when:
─────────────────────                 ──────────────────────
- High selectivity queries            - Write-heavy tables (INSERT/UPDATE/DELETE)
  (few rows match)                       → each write must update all indexes
- Frequent reads on indexed columns    - Low selectivity (e.g., boolean column
- JOIN conditions                        with 50% true / 50% false)
- ORDER BY / GROUP BY                  - Small tables (full table scan is faster)
- Uniqueness constraints               - Columns that are rarely queried
                                       - Too many indexes → more disk, slower writes
```

**Rule of thumb**: If a query returns more than ~10-15% of the table, a full
table scan is often faster than an index scan.

### 3.6 EXPLAIN / EXPLAIN ANALYZE

```sql
-- EXPLAIN: shows the query plan WITHOUT executing it
EXPLAIN
SELECT * FROM users WHERE email = 'alice@example.com';
-- Output example:
--   Index Scan using idx_users_email on users
--     Index Cond: (email = 'alice@example.com'::text)

-- EXPLAIN ANALYZE: actually EXECUTES the query and shows real timing
EXPLAIN ANALYZE
SELECT * FROM orders WHERE user_id = 42 AND status = 'shipped';
-- Output example:
--   Index Scan using idx_orders_user_status on orders
--     Index Cond: (user_id = 42 AND status = 'shipped')
--     Rows Removed by Filter: 0
--     Planning Time: 0.12 ms
--     Execution Time: 0.04 ms
```

**Key things to look for in EXPLAIN output:**

| Scan Type | Meaning | Speed |
|-----------|---------|-------|
| Seq Scan | Full table scan (no index used) | Slow on large tables |
| Index Scan | Uses index, then fetches heap rows | Fast |
| Index Only Scan | Uses covering index, no heap access | Fastest |
| Bitmap Index Scan | Index scan → bitmap → heap fetch | Good for moderate selectivity |
| Nested Loop | For each row in A, scan B | Good for small sets |
| Hash Join | Build hash table of one side | Good for large equi-joins |
| Merge Join | Merge two sorted streams | Good for pre-sorted data |

### 3.7 Partial Indexes

A partial index indexes only a subset of rows. Smaller, faster, and cheaper.

```sql
-- Only index active users (if 90% are inactive, this is much smaller)
CREATE INDEX idx_active_users ON users (email)
WHERE is_active = TRUE;

-- Only useful for queries that include the same WHERE condition:
SELECT * FROM users WHERE email = 'x@y.com' AND is_active = TRUE;  -- uses index
SELECT * FROM users WHERE email = 'x@y.com';                        -- does NOT use index
```

---

## 4. ACID Properties

### 4.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      ACID PROPERTIES                            │
├─────────────────┬───────────────────────────────────────────────┤
│                 │                                               │
│  Atomicity      │  A transaction is ALL or NOTHING.             │
│                 │  If any part fails, the whole thing rolls     │
│                 │  back as if nothing happened.                 │
│                 │                                               │
├─────────────────┼───────────────────────────────────────────────┤
│                 │                                               │
│  Consistency    │  A transaction moves the database from one    │
│                 │  valid state to another. All constraints,     │
│                 │  foreign keys, and checks are satisfied.      │
│                 │                                               │
├─────────────────┼───────────────────────────────────────────────┤
│                 │                                               │
│  Isolation      │  Concurrent transactions don't interfere.     │
│                 │  Each appears to run in isolation, even if    │
│                 │  they execute simultaneously.                 │
│                 │                                               │
├─────────────────┼───────────────────────────────────────────────┤
│                 │                                               │
│  Durability     │  Once committed, the data survives crashes,   │
│                 │  power outages, and hardware failures         │
│                 │  (via WAL / write-ahead logging).             │
│                 │                                               │
└─────────────────┴───────────────────────────────────────────────┘
```

### 4.2 Atomicity Example

```sql
-- Transfer $500 from account A to account B
BEGIN;
    UPDATE accounts SET balance = balance - 500 WHERE id = 'A';
    UPDATE accounts SET balance = balance + 500 WHERE id = 'B';
COMMIT;

-- If the second UPDATE fails (e.g., constraint violation), the first
-- UPDATE is rolled back too. The $500 is never "lost."
```

```
Timeline:
  ┌────────────────────────────────────────────┐
  │ BEGIN                                      │
  │   UPDATE A: balance 1000 → 500    ✓        │
  │   UPDATE B: balance 200 → 700     ✗ FAIL   │
  │ ROLLBACK (automatic)                       │
  │   A goes back to 1000                      │
  │   B stays at 200                           │
  │ Net effect: NOTHING changed                │
  └────────────────────────────────────────────┘
```

### 4.3 Consistency Example

```sql
-- The "balance >= 0" CHECK constraint ensures consistency
ALTER TABLE accounts ADD CONSTRAINT positive_balance CHECK (balance >= 0);

BEGIN;
    UPDATE accounts SET balance = balance - 1500 WHERE id = 'A';
    -- If A only has $1000, this violates the constraint → transaction aborts
    -- Database remains in a CONSISTENT state (no negative balances)
COMMIT;
```

### 4.4 Isolation Example

```
Two concurrent transactions:

  T1: Transfer $100 from A to B
  T2: Read balance of A and B (compute total)

WITHOUT isolation:
  T1: UPDATE A: 1000 → 900
                                 T2: SELECT A → 900
                                 T2: SELECT B → 500  (not yet updated)
  T1: UPDATE B: 500 → 600       T2: total = 1400  ← WRONG! ($100 vanished)
  T1: COMMIT

WITH proper isolation (SERIALIZABLE):
  T2 sees either the state BEFORE T1 or AFTER T1, never in-between.
  BEFORE: A=1000, B=500, total=1500  ✓
  AFTER:  A=900,  B=600, total=1500  ✓
```

### 4.5 Durability Example

```
Timeline:
  1. BEGIN
  2. INSERT INTO orders VALUES (...)
  3. COMMIT  → database writes to WAL (Write-Ahead Log) on disk
  4. *** POWER FAILURE ***
  5. Database restarts, replays WAL
  6. The committed INSERT is still there  ✓

  If the crash happened BEFORE step 3 (COMMIT), the INSERT is lost.
  If the crash happened AFTER step 3, the INSERT survives.
```

### 4.6 Transaction Isolation Levels

```
┌────────────────────┬─────────────┬──────────────────┬─────────────────┐
│ Isolation Level    │ Dirty Reads │ Non-Repeatable   │ Phantom Reads   │
│                    │             │ Reads            │                 │
├────────────────────┼─────────────┼──────────────────┼─────────────────┤
│ READ UNCOMMITTED   │ Possible    │ Possible         │ Possible        │
│ READ COMMITTED     │ Prevented   │ Possible         │ Possible        │
│ REPEATABLE READ    │ Prevented   │ Prevented        │ Possible*       │
│ SERIALIZABLE       │ Prevented   │ Prevented        │ Prevented       │
└────────────────────┴─────────────┴──────────────────┴─────────────────┘

* In PostgreSQL, REPEATABLE READ also prevents phantom reads (it uses
  snapshot isolation / MVCC), making it stricter than the SQL standard.

Default isolation level:
  - PostgreSQL: READ COMMITTED
  - MySQL (InnoDB): REPEATABLE READ
```

```sql
-- Set isolation level for a transaction
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
    -- ... your queries here ...
COMMIT;

-- Set default for session
SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL READ COMMITTED;
```

### 4.7 Common Concurrency Problems (Visual Timelines)

**Dirty Read**: Reading uncommitted data from another transaction.

```
  T1                              T2
  ─────────────────────           ─────────────────────
  BEGIN                           BEGIN
  UPDATE users SET                   │
    name='Bob' WHERE id=1            │
        │                         SELECT name FROM users
        │                           WHERE id=1
        │                           → reads 'Bob' ← DIRTY READ
  ROLLBACK                           │
  (name goes back to 'Alice')     ... uses 'Bob' which never existed!
```

**Non-Repeatable Read**: Same query returns different data within one transaction.

```
  T1                              T2
  ─────────────────────           ─────────────────────
  BEGIN                           BEGIN
  SELECT balance FROM accounts       │
    WHERE id=1 → 1000               │
        │                         UPDATE accounts SET
        │                           balance=500 WHERE id=1
        │                         COMMIT
  SELECT balance FROM accounts       │
    WHERE id=1 → 500  ← DIFFERENT!  │
  (Same query, different result)
```

**Phantom Read**: A query returns different **rows** (new rows appear or disappear).

```
  T1                              T2
  ─────────────────────           ─────────────────────
  BEGIN                           BEGIN
  SELECT COUNT(*) FROM orders        │
    WHERE status='pending' → 5       │
        │                         INSERT INTO orders
        │                           (status) VALUES ('pending')
        │                         COMMIT
  SELECT COUNT(*) FROM orders        │
    WHERE status='pending' → 6       │
  (A new "phantom" row appeared)
```

---

## 5. Sharding

### 5.1 Horizontal vs Vertical Partitioning

```
VERTICAL PARTITIONING (splitting columns):

  Original users table:
  ┌────┬──────┬───────┬───────────────────┬──────────────────────────┐
  │ id │ name │ email │ profile_photo_url │ bio (large text)         │
  └────┴──────┴───────┴───────────────────┴──────────────────────────┘

  Split into:
  users_core:              users_profile:
  ┌────┬──────┬───────┐    ┌────┬───────────────────┬─────────────────┐
  │ id │ name │ email │    │ id │ profile_photo_url │ bio             │
  └────┴──────┴───────┘    └────┴───────────────────┴─────────────────┘
  (hot data, fast access)  (cold data, infrequent access)


HORIZONTAL PARTITIONING / SHARDING (splitting rows):

  Original users table (10M rows):
  ┌──────────────────────────────────────┐
  │  id=1, id=2, id=3 ... id=10,000,000 │
  └──────────────────────────────────────┘

  Sharded into 3 databases:
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │   Shard 0   │  │   Shard 1   │  │   Shard 2   │
  │ id%3 == 0   │  │ id%3 == 1   │  │ id%3 == 2   │
  │ 3,6,9,12... │  │ 1,4,7,10... │  │ 2,5,8,11... │
  │  ~3.33M     │  │  ~3.33M     │  │  ~3.33M     │
  └─────────────┘  └─────────────┘  └─────────────┘
    Server A          Server B          Server C
```

### 5.2 Sharding Strategies

**1. Hash-Based Sharding**

```
shard_number = hash(shard_key) % number_of_shards

Example: shard_key = user_id
  user_id=1  → hash(1) % 3 = 1 → Shard 1
  user_id=2  → hash(2) % 3 = 2 → Shard 2
  user_id=3  → hash(3) % 3 = 0 → Shard 0

Pros: Even distribution
Cons: Adding/removing shards requires rehashing (consistent hashing helps)
```

**2. Range-Based Sharding**

```
  Shard by user_id range:
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │   Shard A    │  │   Shard B    │  │   Shard C    │
  │ id: 1-1M    │  │ id: 1M-2M   │  │ id: 2M-3M   │
  └──────────────┘  └──────────────┘  └──────────────┘

Pros: Easy to understand, range queries stay on one shard
Cons: Hotspots if recent IDs are more active (e.g., new users)
```

**3. Directory-Based Sharding**

```
  A lookup table maps each key to its shard:
  ┌───────────┬───────┐
  │ user_id   │ shard │
  ├───────────┼───────┤
  │ 1         │ A     │
  │ 2         │ C     │
  │ 3         │ B     │
  │ ...       │ ...   │
  └───────────┴───────┘

Pros: Flexible, can rebalance individual keys
Cons: Lookup table is a single point of failure and bottleneck
```

### 5.3 Sharding Architecture

```
                    ┌──────────────┐
                    │   App Layer  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Shard Router │  ← determines shard from key
                    │  / Proxy      │
                    └──┬───┬───┬───┘
                       │   │   │
            ┌──────────┘   │   └──────────┐
            ▼              ▼              ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │  Shard 0   │ │  Shard 1   │ │  Shard 2   │
     │ (Primary)  │ │ (Primary)  │ │ (Primary)  │
     │     │      │ │     │      │ │     │      │
     │  Replica   │ │  Replica   │ │  Replica   │
     └────────────┘ └────────────┘ └────────────┘
```

### 5.4 Pros and Cons

```
PROS:                                CONS:
─────                                ─────
+ Horizontal scalability             - Increased complexity
+ Each shard handles less data       - Cross-shard JOINs are expensive/impossible
+ Smaller indexes → faster queries   - Distributed transactions are hard
+ Fault isolation (one shard down    - Resharding is painful
  doesn't affect others)             - No global unique constraints
                                     - Aggregation queries need scatter-gather
                                     - Operational overhead (backup, monitoring)
```

### 5.5 Cross-Shard Queries

```
Query: "Find all orders for user_id=42"
  → shard = hash(42) % 3 = 0  → query Shard 0 only  ✓ FAST

Query: "Find the top 10 users by order count"
  → must query ALL shards, aggregate results  → SLOW
  → this is called a "scatter-gather" query

Query: "JOIN users u ON orders.user_id = u.id" where users and orders
  are on different shards:
  → either co-locate the data (shard both by user_id)
  → or fetch from both shards and join in the application layer
```

### 5.6 Resharding Challenges

When you need to change the number of shards (e.g., 3 -> 5):

```
Naive approach:  rehash ALL keys → massive data migration, downtime

Better: Consistent hashing
  - Each shard owns a range on a hash ring
  - Adding a shard only moves keys from adjacent ranges
  - Much less data movement

  Hash Ring (0 to 2^32):
       0
      ╱   ╲
   Shard0   Shard1       Adding Shard3 between 0 and 1
      ╲   ╱              only moves some keys from Shard1
      Shard2

Even better: Virtual shards / logical shards
  - Create 256 logical shards, map them to 3 physical servers
  - To add a 4th server, just move some logical shards
  - No rehashing needed
```

---

## 6. Database Design

### 6.1 Normalization

**1NF (First Normal Form)**: Each column holds atomic (indivisible) values.

```
VIOLATION of 1NF:
┌────┬──────┬───────────────────┐
│ id │ name │ phone_numbers     │
├────┼──────┼───────────────────┤
│  1 │ Alice│ 555-1234,555-5678 │  ← multiple values in one column!
└────┴──────┴───────────────────┘

FIXED (1NF):
┌────┬──────┬──────────┐
│ id │ name │ phone    │
├────┼──────┼──────────┤
│  1 │ Alice│ 555-1234 │
│  1 │ Alice│ 555-5678 │    or better: separate phone_numbers table
└────┴──────┴──────────┘
```

**2NF (Second Normal Form)**: 1NF + no partial dependencies on a composite key.

```
VIOLATION of 2NF (composite key: student_id + course_id):
┌────────────┬───────────┬────────────┬──────────────┐
│ student_id │ course_id │ grade      │ student_name │  ← depends only on student_id
└────────────┴───────────┴────────────┴──────────────┘
  student_name depends only on student_id, not the full key.

FIXED (2NF):
students: (student_id, student_name)
enrollments: (student_id, course_id, grade)
```

**3NF (Third Normal Form)**: 2NF + no transitive dependencies.

```
VIOLATION of 3NF:
┌────┬──────┬──────────────┬───────────┐
│ id │ name │ department   │ dept_head │  ← dept_head depends on department,
└────┴──────┴──────────────┴───────────┘     not directly on id

FIXED (3NF):
employees:   (id, name, department_id)
departments: (department_id, department_name, dept_head)
```

### 6.2 Denormalization

**When to denormalize:**
- Read-heavy workloads where JOINs are too expensive
- Caching computed values to avoid repeated aggregations
- Materialized views as a middle ground

```sql
-- Normalized: need a JOIN to get order totals
SELECT u.name, SUM(o.amount) AS total
FROM users u
JOIN orders o ON u.id = o.user_id
GROUP BY u.name;

-- Denormalized: store total_spent directly on users table
ALTER TABLE users ADD COLUMN total_spent DECIMAL DEFAULT 0;

-- Update it via trigger or application code
-- Faster reads but data can become inconsistent if not updated correctly
```

### 6.3 Foreign Keys & Constraints

```sql
CREATE TABLE orders (
    id         SERIAL PRIMARY KEY,
    user_id    INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    amount     DECIMAL(10,2) NOT NULL CHECK (amount > 0),
    status     VARCHAR(20) NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending', 'shipped', 'delivered', 'cancelled')),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Named constraint for better error messages
    CONSTRAINT fk_orders_user FOREIGN KEY (user_id) REFERENCES users(id)
);

-- ON DELETE options:
--   CASCADE:     delete orders when user is deleted
--   SET NULL:    set user_id to NULL when user is deleted
--   SET DEFAULT: set user_id to its default value
--   RESTRICT:    prevent deletion of user if orders exist (default)
--   NO ACTION:   same as RESTRICT but checked at end of transaction
```

### 6.4 Soft Deletes vs Hard Deletes

```sql
-- HARD DELETE: row is gone forever
DELETE FROM users WHERE id = 42;

-- SOFT DELETE: mark as deleted, keep the data
ALTER TABLE users ADD COLUMN deleted_at TIMESTAMP NULL;

UPDATE users SET deleted_at = NOW() WHERE id = 42;

-- All queries must filter:
SELECT * FROM users WHERE deleted_at IS NULL;

-- Or use a view:
CREATE VIEW active_users AS
SELECT * FROM users WHERE deleted_at IS NULL;
```

```
SOFT DELETES                         HARD DELETES
─────────────                        ──────────────
+ Data recovery is trivial           + Simpler queries (no WHERE filter)
+ Audit trail                        + Less storage
+ Referential integrity preserved    + Better performance
- Every query needs WHERE filter     - Data is gone (unless you have backups)
- Table grows forever                - Foreign key issues on delete
- Unique constraints get tricky      - No audit trail
  (use partial unique index on
   WHERE deleted_at IS NULL)
```

---

## 7. Performance Optimization

### 7.1 Query Optimization Techniques

```sql
-- 1. SELECT only what you need (avoid SELECT *)
SELECT id, name, email FROM users;  -- not SELECT *

-- 2. Use indexes wisely
CREATE INDEX idx_orders_user_date ON orders (user_id, created_at DESC);

-- 3. Avoid functions on indexed columns
WHERE created_at >= '2025-01-01'           -- ✓ uses index
WHERE EXTRACT(YEAR FROM created_at) = 2025 -- ✗ index not used (function on column)

-- 4. Use EXISTS instead of IN for large subqueries
WHERE EXISTS (SELECT 1 FROM orders WHERE orders.user_id = users.id)  -- ✓
WHERE id IN (SELECT user_id FROM orders)                              -- slower

-- 5. Use LIMIT for pagination (but be aware of deep pagination problems)
SELECT * FROM orders ORDER BY id LIMIT 20 OFFSET 10000;  -- slow (scans 10020 rows)
SELECT * FROM orders WHERE id > 10000 ORDER BY id LIMIT 20;  -- fast (keyset pagination)

-- 6. Batch operations
INSERT INTO logs (message) VALUES ('a'), ('b'), ('c');  -- one round trip
-- instead of three separate INSERTs

-- 7. Use UNION ALL instead of UNION when you don't need deduplication
SELECT id FROM table_a
UNION ALL          -- ✓ no sort/dedup needed
SELECT id FROM table_b;
```

### 7.2 The N+1 Query Problem

```
PROBLEM:
  1 query to get all users:
    SELECT * FROM users;  → returns 100 users

  Then 100 queries, one per user:
    SELECT * FROM orders WHERE user_id = 1;
    SELECT * FROM orders WHERE user_id = 2;
    ...
    SELECT * FROM orders WHERE user_id = 100;

  Total: 101 queries!  (1 + N where N=100)

SOLUTIONS:

  1. JOIN (one query):
     SELECT u.*, o.*
     FROM users u
     LEFT JOIN orders o ON u.id = o.user_id;

  2. Eager loading / batch fetch:
     SELECT * FROM users;
     SELECT * FROM orders WHERE user_id IN (1, 2, 3, ..., 100);
     → 2 queries total

  3. ORM solutions:
     # SQLAlchemy: use joinedload or subqueryload
     session.query(User).options(joinedload(User.orders)).all()
```

### 7.3 Connection Pooling

```
WITHOUT pooling:                    WITH pooling:
  App → connect → query → close       App → get from pool → query → return to pool
  App → connect → query → close       App → get from pool → query → return to pool
  (TCP handshake + auth each time)    (connections are reused)

  Connection lifecycle:                Pool:
  ~50-100ms per connection setup       ┌─────────────────────────┐
                                       │  Pool (e.g., 20 conns)  │
                                       │  ┌──┐┌──┐┌──┐┌──┐      │
                                       │  │C1││C2││C3││C4│ ...   │
                                       │  └──┘└──┘└──┘└──┘      │
                                       └─────────────────────────┘

  Tools:
  - PgBouncer (external, for PostgreSQL)
  - SQLAlchemy connection pool (built-in)
  - asyncpg pool (for async Python)
```

### 7.4 Read Replicas

```
  ┌──────────────┐        Replication        ┌────────────────┐
  │   Primary    │ ──────────────────────────▶│   Replica 1    │
  │  (read+write)│            │               │  (read-only)   │
  └──────────────┘            │               └────────────────┘
                              │
                              └──────────────▶┌────────────────┐
                                              │   Replica 2    │
                                              │  (read-only)   │
                                              └────────────────┘

  - Writes go to Primary
  - Reads are distributed across Replicas
  - Eventual consistency: replicas may lag behind Primary by milliseconds
  - Good for read-heavy workloads (90% reads / 10% writes)

  Caveats:
  - Replication lag can cause stale reads
  - After a write, read from Primary (read-your-writes consistency)
```

### 7.5 Materialized Views

```sql
-- A materialized view is a cached result of a query, stored as a table.
CREATE MATERIALIZED VIEW mv_department_stats AS
SELECT
    department,
    COUNT(*) AS employee_count,
    AVG(salary) AS avg_salary,
    MAX(salary) AS max_salary
FROM employees
GROUP BY department;

-- Query it like a table (instant, no re-computation):
SELECT * FROM mv_department_stats WHERE department = 'Engineering';

-- Refresh it periodically:
REFRESH MATERIALIZED VIEW mv_department_stats;

-- Refresh concurrently (no lock, but requires a unique index):
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_department_stats;

-- Trade-off: fast reads, but data can be stale between refreshes.
```

### 7.6 Caching Strategies

```
Application-level caching with Redis/Memcached:

  ┌───────┐     ┌─────────┐     ┌──────────────┐
  │  App  │────▶│  Cache  │────▶│   Database   │
  └───────┘     │ (Redis) │     └──────────────┘
                └─────────┘

  Read pattern (Cache-Aside):
    1. Check cache for key
    2. If cache HIT → return cached data
    3. If cache MISS → query database → store in cache → return

  Write patterns:
    - Write-through: write to cache AND database simultaneously
    - Write-behind: write to cache, async flush to database
    - Cache invalidation: write to database, delete cache key

  Common pitfalls:
    - Cache stampede: many requests miss cache simultaneously
      → use locking or request coalescing
    - Stale data: cache not updated after database write
      → set TTL (time-to-live) on cache keys
```

---

## 8. PostgreSQL Specifics

### 8.1 JSONB Columns

```sql
CREATE TABLE events (
    id    SERIAL PRIMARY KEY,
    data  JSONB NOT NULL
);

INSERT INTO events (data) VALUES
('{"type": "click", "page": "/home", "user": {"id": 42, "name": "Alice"}}');

-- Query JSONB fields
SELECT data->>'type' AS event_type              -- text extraction
FROM events
WHERE data->>'type' = 'click';

SELECT data->'user'->>'name' AS user_name       -- nested extraction
FROM events;

-- Check if key exists
SELECT * FROM events WHERE data ? 'type';

-- Check if JSONB contains a subset
SELECT * FROM events
WHERE data @> '{"type": "click"}';              -- containment operator

-- Index JSONB for fast queries
CREATE INDEX idx_events_data ON events USING gin (data);

-- Index a specific JSONB path
CREATE INDEX idx_events_type ON events ((data->>'type'));
```

### 8.2 Array Types

```sql
CREATE TABLE articles (
    id   SERIAL PRIMARY KEY,
    title TEXT,
    tags  TEXT[]                 -- array of text
);

INSERT INTO articles (title, tags)
VALUES ('SQL Guide', ARRAY['sql', 'database', 'postgresql']);

-- Check if array contains a value
SELECT * FROM articles WHERE 'sql' = ANY(tags);

-- Check if array contains all specified values
SELECT * FROM articles WHERE tags @> ARRAY['sql', 'database'];

-- Unnest array into rows
SELECT id, unnest(tags) AS tag FROM articles;

-- Index for array operations
CREATE INDEX idx_articles_tags ON articles USING gin (tags);
```

### 8.3 Full-Text Search

```sql
-- Convert text to tsvector (searchable tokens)
SELECT to_tsvector('english', 'The quick brown fox jumps over the lazy dog');
-- Result: 'brown':3 'dog':9 'fox':4 'jump':5 'lazi':8 'quick':2

-- Convert search query to tsquery
SELECT to_tsquery('english', 'quick & fox');
-- Result: 'quick' & 'fox'

-- Full-text search query
SELECT title, content
FROM articles
WHERE to_tsvector('english', content) @@ to_tsquery('english', 'database & optimization');

-- Add a generated tsvector column and index for performance
ALTER TABLE articles ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || content)) STORED;

CREATE INDEX idx_articles_search ON articles USING gin (search_vector);

-- Now queries are fast:
SELECT title FROM articles
WHERE search_vector @@ to_tsquery('english', 'database & optimization');

-- Ranking results
SELECT title,
       ts_rank(search_vector, to_tsquery('english', 'database')) AS rank
FROM articles
WHERE search_vector @@ to_tsquery('english', 'database')
ORDER BY rank DESC;
```

### 8.4 Advisory Locks

Application-level locks that don't lock any actual table rows.

```sql
-- Acquire an advisory lock (blocks until available)
SELECT pg_advisory_lock(12345);

-- Do work that needs mutual exclusion...

-- Release the lock
SELECT pg_advisory_unlock(12345);

-- Try to acquire (non-blocking, returns true/false)
SELECT pg_try_advisory_lock(12345);

-- Session-level vs transaction-level:
SELECT pg_advisory_lock(12345);              -- held until explicitly released or session ends
SELECT pg_advisory_xact_lock(12345);         -- released at end of transaction

-- Use case: prevent duplicate cron job execution
-- Worker 1: SELECT pg_try_advisory_lock(42) → true  (runs the job)
-- Worker 2: SELECT pg_try_advisory_lock(42) → false (skips)
```

### 8.5 LISTEN / NOTIFY

Lightweight pub/sub built into PostgreSQL, useful for real-time notifications.

```sql
-- Session 1: listen for events
LISTEN order_updates;

-- Session 2: send a notification
NOTIFY order_updates, '{"order_id": 123, "status": "shipped"}';

-- Session 1 receives:
-- Asynchronous notification "order_updates" with payload
-- '{"order_id": 123, "status": "shipped"}' received from server process.

-- In Python with asyncpg:
-- await connection.add_listener('order_updates', callback_fn)
```

### 8.6 Useful Extensions

```sql
-- pg_trgm: trigram-based fuzzy text matching
CREATE EXTENSION pg_trgm;
CREATE INDEX idx_users_name_trgm ON users USING gin (name gin_trgm_ops);

SELECT * FROM users
WHERE name % 'Alce';       -- fuzzy match (finds "Alice")
-- similarity('Alce', 'Alice') = 0.4

SELECT name, similarity(name, 'Alce') AS sim
FROM users
WHERE name % 'Alce'
ORDER BY sim DESC;

-- pgvector: vector similarity search (critical for AI/ML applications)
CREATE EXTENSION vector;

CREATE TABLE embeddings (
    id        SERIAL PRIMARY KEY,
    content   TEXT,
    embedding vector(1536)        -- OpenAI ada-002 dimension
);

-- Insert an embedding
INSERT INTO embeddings (content, embedding)
VALUES ('Hello world', '[0.1, 0.2, ..., 0.5]');

-- Nearest neighbor search (cosine distance)
SELECT content, embedding <=> '[0.1, 0.2, ..., 0.5]' AS distance
FROM embeddings
ORDER BY embedding <=> '[0.1, 0.2, ..., 0.5]'
LIMIT 10;

-- Create an index for fast approximate nearest neighbor search
CREATE INDEX idx_embeddings ON embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- Or use HNSW for better recall:
CREATE INDEX idx_embeddings_hnsw ON embeddings
USING hnsw (embedding vector_cosine_ops);
```

---

## 9. SQL in Python

### 9.1 SQLAlchemy: Core vs ORM

```python
# ================================================================
# SQLAlchemy CORE — write SQL-like expressions in Python
# ================================================================
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, select

engine = create_engine("postgresql://user:pass@localhost/mydb")
metadata = MetaData()

users = Table("users", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(100)),
    Column("email", String(200)),
)

# Query using Core expression language
stmt = select(users).where(users.c.name == "Alice")
with engine.connect() as conn:
    result = conn.execute(stmt)
    for row in result:
        print(row.id, row.name, row.email)


# ================================================================
# SQLAlchemy ORM — map Python classes to tables
# ================================================================
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, relationship
from sqlalchemy import ForeignKey

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id:    Mapped[int] = mapped_column(primary_key=True)
    name:  Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(200))

    orders: Mapped[list["Order"]] = relationship(back_populates="user")

class Order(Base):
    __tablename__ = "orders"

    id:      Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    amount:  Mapped[float]

    user: Mapped["User"] = relationship(back_populates="orders")

# Querying with ORM
with Session(engine) as session:
    # Simple query
    user = session.query(User).filter_by(name="Alice").first()

    # Modern style (SQLAlchemy 2.0)
    stmt = select(User).where(User.name == "Alice")
    user = session.scalars(stmt).first()

    # Eager loading to avoid N+1
    from sqlalchemy.orm import joinedload
    stmt = select(User).options(joinedload(User.orders))
    users = session.scalars(stmt).unique().all()

    # Create
    new_user = User(name="Bob", email="bob@example.com")
    session.add(new_user)
    session.commit()
```

**Core vs ORM — When to Use Which:**

```
Core:                                ORM:
─────                                ────
- Bulk operations (fast)             - CRUD with business logic
- Complex queries / reporting        - Relationships between models
- ETL / data pipelines               - Validation, lifecycle hooks
- When you want SQL-level control    - When you want Pythonic abstractions
- Slightly faster (no object map)    - Better developer experience
```

### 9.2 Alembic Migrations

```python
# Initialize Alembic in your project:
#   alembic init alembic

# Create a migration:
#   alembic revision --autogenerate -m "add users table"

# Generated migration file (alembic/versions/xxxx_add_users_table.py):
def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("email", sa.String(200), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("idx_users_email", "users", ["email"])

def downgrade():
    op.drop_index("idx_users_email")
    op.drop_table("users")

# Run migrations:
#   alembic upgrade head       # apply all pending migrations
#   alembic downgrade -1       # rollback last migration
#   alembic history            # show migration history
#   alembic current            # show current version

# Best practices:
# - Always test both upgrade() and downgrade()
# - Never edit a migration that has been applied to production
# - Use --autogenerate but ALWAYS review the generated code
# - Keep migrations small and focused
# - Add data migrations as separate steps from schema migrations
```

### 9.3 asyncpg for Async PostgreSQL

```python
import asyncio
import asyncpg

async def main():
    # Create a connection pool
    pool = await asyncpg.create_pool(
        "postgresql://user:pass@localhost/mydb",
        min_size=5,
        max_size=20,
    )

    # Use the pool
    async with pool.acquire() as conn:
        # Simple query
        row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", 42)
        print(row["name"], row["email"])

        # Fetch multiple rows
        rows = await conn.fetch(
            "SELECT * FROM users WHERE created_at > $1",
            datetime(2025, 1, 1),
        )

        # Execute (no return value)
        await conn.execute(
            "INSERT INTO users (name, email) VALUES ($1, $2)",
            "Alice", "alice@example.com",
        )

        # Transaction
        async with conn.transaction():
            await conn.execute("UPDATE accounts SET balance = balance - 100 WHERE id = $1", 1)
            await conn.execute("UPDATE accounts SET balance = balance + 100 WHERE id = $1", 2)

        # Prepared statements (faster for repeated queries)
        stmt = await conn.prepare("SELECT * FROM users WHERE id = $1")
        user = await stmt.fetchrow(42)

        # LISTEN/NOTIFY
        await conn.add_listener("order_updates", lambda conn, pid, channel, payload:
            print(f"Notification: {payload}"))

    await pool.close()

asyncio.run(main())
```

**Why asyncpg over psycopg2?**

```
asyncpg:                              psycopg2:
────────                              ─────────
- Async/await native                  - Synchronous (blocking)
- 3-5x faster than psycopg2          - Widely used, mature
- Binary protocol (fewer bytes)       - Text protocol
- Built-in connection pooling         - Needs external pool (or psycopg2.pool)
- Great with FastAPI / aiohttp        - Great with Flask / Django

psycopg3 (psycopg):
─────────
- Both sync and async
- Modern, actively developed
- Binary protocol support
- Pipeline mode for batching
```

### 9.4 Connection Pooling in Python

```python
# SQLAlchemy built-in pooling
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://user:pass@localhost/mydb",
    pool_size=20,           # number of permanent connections
    max_overflow=10,        # extra connections when pool is full
    pool_timeout=30,        # seconds to wait for a connection
    pool_recycle=1800,      # recycle connections after 30 minutes
    pool_pre_ping=True,     # test connections before use (handles stale connections)
)

# asyncpg pool (for async code)
pool = await asyncpg.create_pool(
    dsn="postgresql://user:pass@localhost/mydb",
    min_size=10,            # minimum connections to keep open
    max_size=20,            # maximum connections
    max_inactive_connection_lifetime=300,  # close idle connections after 5 min
)

# External pooler: PgBouncer
# - Sits between app and PostgreSQL
# - Modes: session (1:1), transaction (shared), statement (most aggressive)
# - Handles thousands of app connections with fewer DB connections
# - Config: pgbouncer.ini
#     [databases]
#     mydb = host=127.0.0.1 port=5432 dbname=mydb
#     [pgbouncer]
#     pool_mode = transaction
#     default_pool_size = 20
#     max_client_conn = 1000
```

---

## 10. Q&A Section

### Q1: Explain the difference between INNER JOIN and LEFT JOIN.

**INNER JOIN** returns only rows that have matching values in **both** tables.
If a row in the left table has no match in the right table, it is excluded.

**LEFT JOIN** returns **all** rows from the left table, plus matching rows from
the right table. If there is no match, the right-side columns are filled with
NULL.

```sql
-- INNER JOIN: users WITHOUT orders are excluded
SELECT u.name, o.id
FROM users u INNER JOIN orders o ON u.id = o.user_id;

-- LEFT JOIN: users WITHOUT orders appear with NULL order_id
SELECT u.name, o.id
FROM users u LEFT JOIN orders o ON u.id = o.user_id;
```

**When to use which:**
- INNER JOIN when you only care about rows with matches on both sides.
- LEFT JOIN when you want all rows from the left table regardless of matches.

---

### Q2: What is a composite index and how does column order matter?

A composite index is an index on **two or more columns**. The columns are
ordered, and the index follows the **leftmost prefix rule**.

For an index on `(A, B, C)`:
- The data is sorted first by A, then by B within each A value, then by C.
- Queries filtering on `A` alone, `A + B`, or `A + B + C` can use the index.
- Queries filtering on `B` alone or `C` alone **cannot** use the index.

**Column order guideline:**
1. Put equality conditions first (`WHERE A = ?`).
2. Then range conditions (`WHERE B > ?`).
3. Then columns used in ORDER BY or SELECT (for covering indexes).

---

### Q3: Explain ACID properties with examples.

- **Atomicity**: A bank transfer either debits AND credits, or does neither.
  If the credit fails, the debit is rolled back.
- **Consistency**: A CHECK constraint `balance >= 0` is never violated, even
  during concurrent transactions.
- **Isolation**: Two concurrent transfers don't interfere. Each sees a
  consistent snapshot of the database.
- **Durability**: Once a COMMIT returns successfully, the data survives a
  server crash because it has been written to the WAL (Write-Ahead Log).

(See Section 4 for detailed visual examples of each.)

---

### Q4: What are transaction isolation levels? Name the problems each one prevents.

| Level | Dirty Read | Non-Repeatable Read | Phantom Read |
|-------|-----------|-------------------|-------------|
| READ UNCOMMITTED | Possible | Possible | Possible |
| READ COMMITTED | Prevented | Possible | Possible |
| REPEATABLE READ | Prevented | Prevented | Possible* |
| SERIALIZABLE | Prevented | Prevented | Prevented |

*PostgreSQL's REPEATABLE READ also prevents phantoms via snapshot isolation.

Higher isolation = more correctness but lower concurrency/throughput.
Most applications use READ COMMITTED (PostgreSQL default).

---

### Q5: How does sharding work? When would you use it?

Sharding splits a large table across multiple database servers (shards), each
holding a subset of rows. A **shard key** (e.g., user_id) determines which
shard a row belongs to.

**When to shard:**
- Single database cannot handle the write load.
- Dataset is too large to fit on one server.
- You need geographic data locality.

**When NOT to shard:**
- Read replicas can handle the load (simpler).
- Vertical scaling (bigger server) is still feasible.
- The data is small enough for one database.

Sharding introduces complexity: cross-shard joins, distributed transactions,
and resharding are all hard problems.

---

### Q6: What is the N+1 query problem and how to solve it?

The N+1 problem occurs when code executes 1 query to fetch a list of N
parent records, then N additional queries to fetch related data for each
parent. This results in N+1 total queries instead of 1-2.

**Solutions:**
1. **JOIN**: Fetch everything in one query.
2. **Batch fetch**: `WHERE id IN (...)` to get all related records at once.
3. **ORM eager loading**: `joinedload()` or `subqueryload()` in SQLAlchemy.

```python
# BAD: N+1
users = session.query(User).all()             # 1 query
for user in users:
    print(user.orders)                         # N queries (one per user)

# GOOD: Eager loading (2 queries total)
users = session.query(User).options(
    subqueryload(User.orders)
).all()
```

---

### Q7: Explain window functions with an example.

Window functions compute a value for each row based on a set of related rows
(the "window"), without collapsing the result set like GROUP BY does.

```sql
-- Get each employee's salary and how it compares to their department average
SELECT
    name,
    department,
    salary,
    AVG(salary) OVER (PARTITION BY department) AS dept_avg,
    salary - AVG(salary) OVER (PARTITION BY department) AS diff_from_avg
FROM employees;
```

Common window functions: `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`, `LAG()`,
`LEAD()`, `SUM() OVER`, `AVG() OVER`, `NTILE()`, `FIRST_VALUE()`,
`LAST_VALUE()`.

---

### Q8: What is a covering index?

A covering index includes **all columns** referenced by a query (in the WHERE,
SELECT, ORDER BY, etc.). This allows the database to satisfy the query using
**only the index**, without accessing the table data (the "heap").

```sql
-- Query needs: name (filter) and email (select)
SELECT email FROM users WHERE name = 'Alice';

-- Covering index using INCLUDE:
CREATE INDEX idx ON users (name) INCLUDE (email);

-- EXPLAIN shows "Index Only Scan" → no heap access → faster
```

---

### Q9: How does EXPLAIN ANALYZE help optimize queries?

`EXPLAIN` shows the **planned** execution strategy without running the query.
`EXPLAIN ANALYZE` actually **executes** the query and shows real timings.

**What to look for:**
- **Seq Scan** on large tables → consider adding an index.
- **Nested Loop** with large inputs → may need Hash Join or optimization.
- **Estimated rows** vs **actual rows** → large difference means stale statistics
  (run `ANALYZE table_name;`).
- **Sort** operations → can sometimes be eliminated with an index.

```sql
EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 42;
-- Look at: Planning Time, Execution Time, actual rows, scan type
```

---

### Q10: When would you denormalize a database?

Denormalize when:
- **Read performance** is critical and JOINs are too slow.
- Data is **read far more than written** (e.g., 100:1 ratio).
- You need **precomputed aggregates** (e.g., `total_orders` on user row).
- You're using a **NoSQL** store that doesn't support JOINs.

Downsides: data duplication, risk of inconsistency, more complex write logic.

A **materialized view** is a good middle ground: denormalized for reads, but
refreshed from normalized source data.

---

### Q11: What is a deadlock and how to prevent it?

A deadlock occurs when two or more transactions are waiting for each other's
locks, forming a cycle.

```
T1: LOCK row A, then tries to LOCK row B  ← waiting for T2
T2: LOCK row B, then tries to LOCK row A  ← waiting for T1
→ Neither can proceed = DEADLOCK
```

**Prevention:**
1. Always acquire locks in the **same order** (e.g., by ascending ID).
2. Keep transactions **short** (less time holding locks).
3. Use **SELECT ... FOR UPDATE** with `NOWAIT` or `SKIP LOCKED`.
4. The database auto-detects deadlocks and kills one transaction
   (you must handle the error and retry).

```sql
-- Lock ordering: always lock the lower ID first
BEGIN;
SELECT * FROM accounts WHERE id = LEAST(1, 2) FOR UPDATE;
SELECT * FROM accounts WHERE id = GREATEST(1, 2) FOR UPDATE;
-- ... do work ...
COMMIT;
```

---

### Q12: Difference between DELETE, TRUNCATE, and DROP?

| | DELETE | TRUNCATE | DROP |
|---|--------|----------|------|
| **What** | Removes specific rows | Removes all rows | Removes the entire table |
| **WHERE** | Can filter rows | No (removes ALL) | N/A |
| **Logged** | Row-by-row (slow for many rows) | Minimal logging (fast) | Minimal logging |
| **Rollback** | Yes (in a transaction) | Yes in PostgreSQL, not in MySQL | Yes in PostgreSQL |
| **Triggers** | Fires row-level triggers | Does NOT fire row triggers | N/A |
| **Identity** | Does not reset sequence | Resets sequence (with RESTART IDENTITY) | N/A |
| **Space** | Does not free disk immediately | Frees disk immediately | Frees disk |

```sql
DELETE FROM users WHERE id = 42;          -- remove one row
TRUNCATE TABLE logs;                      -- remove all rows, fast
TRUNCATE TABLE logs RESTART IDENTITY;     -- also reset auto-increment
DROP TABLE IF EXISTS temp_data;           -- remove table entirely
```

---

### Q13: How do you handle migrations in production?

1. **Never run destructive migrations without a plan.** Column drops, type
   changes, and table renames can break running application instances.

2. **Use a migration tool** (Alembic, Flyway, Liquibase) for version control.

3. **Expand-and-contract pattern** for zero-downtime changes:
   - Phase 1 (expand): Add new column, deploy code that writes to both.
   - Phase 2 (migrate): Backfill old data to new column.
   - Phase 3 (contract): Remove old column, deploy code that uses only new.

4. **Always test rollback** (`downgrade`) in staging.

5. **Large data migrations** should run in batches to avoid long locks:
   ```sql
   -- Instead of one huge UPDATE:
   UPDATE users SET new_col = old_col;  -- locks table for minutes

   -- Batch it:
   UPDATE users SET new_col = old_col WHERE id BETWEEN 1 AND 10000;
   UPDATE users SET new_col = old_col WHERE id BETWEEN 10001 AND 20000;
   ```

6. **Add indexes concurrently** in PostgreSQL:
   ```sql
   CREATE INDEX CONCURRENTLY idx_name ON table (col);
   -- Does not block writes (but takes longer)
   ```

---

### Q14: What is the difference between WHERE and HAVING?

- **WHERE** filters individual **rows** before grouping. It cannot reference
  aggregate functions.
- **HAVING** filters **groups** after GROUP BY. It can reference aggregates.

```sql
SELECT department, COUNT(*) AS cnt
FROM employees
WHERE salary > 50000         -- filter rows first (before grouping)
GROUP BY department
HAVING COUNT(*) > 3;         -- then filter groups (after grouping)
```

---

### Q15: Explain the difference between UNION and UNION ALL.

- **UNION**: combines results and **removes duplicates** (requires sort/hash).
- **UNION ALL**: combines results and **keeps all rows** (faster, no dedup).

Always use `UNION ALL` unless you specifically need deduplication, because the
dedup step adds significant cost.

---

### Q16: What are CTEs and when are they useful?

A CTE (Common Table Expression) is a named temporary result set defined with
the `WITH` clause. It exists only for the duration of the query.

**Use cases:**
- Break complex queries into readable steps.
- Reference the same subquery multiple times without repeating it.
- Recursive queries (tree traversal, graph traversal).

```sql
WITH active_users AS (
    SELECT * FROM users WHERE is_active = TRUE
),
user_orders AS (
    SELECT user_id, COUNT(*) AS order_count
    FROM orders
    GROUP BY user_id
)
SELECT au.name, uo.order_count
FROM active_users au
JOIN user_orders uo ON au.id = uo.user_id;
```

**Note:** In PostgreSQL 12+, CTEs are automatically inlined (optimized) unless
you use `WITH ... AS MATERIALIZED` to force materialization.

---

### Q17: How does PostgreSQL MVCC work?

MVCC (Multi-Version Concurrency Control) means PostgreSQL keeps **multiple
versions** of each row. Readers don't block writers, and writers don't block
readers.

- Each transaction gets a **snapshot** of the database at its start time.
- An UPDATE creates a **new version** of the row (the old version is kept).
- Old versions are cleaned up by **VACUUM** once no transaction can see them.
- This is why PostgreSQL doesn't need read locks for SELECT statements.

```
Row versions for user_id=1:
  Version 1: name='Alice'  (created by txn 100, deleted by txn 105)
  Version 2: name='Alicia' (created by txn 105, still active)

  Transaction 103 (started before 105): sees Version 1 ('Alice')
  Transaction 106 (started after 105):  sees Version 2 ('Alicia')
```

---

### Q18: What is a partial index and when would you use one?

A partial index indexes only rows that match a given condition. It is smaller
and faster than a full index.

```sql
CREATE INDEX idx_active_orders ON orders (created_at)
WHERE status = 'pending';
```

**When to use:**
- Most rows don't match the query condition (e.g., 5% pending, 95% completed).
- You frequently query the small subset.
- You want to enforce a conditional uniqueness constraint.

```sql
-- Unique email, but only for non-deleted users
CREATE UNIQUE INDEX idx_unique_active_email ON users (email)
WHERE deleted_at IS NULL;
```

---

### Q19: How would you find and optimize a slow query?

1. **Identify**: Check `pg_stat_statements` for slow queries, or enable
   `log_min_duration_statement` in `postgresql.conf`.

2. **Analyze**: Run `EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)` on the query.

3. **Look for:**
   - Sequential scans on large tables (add an index).
   - High `actual rows` vs `estimated rows` (run `ANALYZE table;`).
   - Nested loops with large inputs (restructure query or add indexes).
   - Sorts that spill to disk (increase `work_mem` or add index).

4. **Common fixes:**
   - Add appropriate indexes (considering column order for composites).
   - Rewrite subqueries as JOINs or CTEs.
   - Add `LIMIT` if only a few rows are needed.
   - Use keyset pagination instead of `OFFSET`.
   - Denormalize or use materialized views for complex aggregations.

---

### Q20: What is the difference between optimistic and pessimistic locking?

**Pessimistic locking**: Lock the row before reading it. Other transactions
wait until the lock is released.

```sql
BEGIN;
SELECT * FROM products WHERE id = 1 FOR UPDATE;  -- lock the row
-- ... update the product ...
UPDATE products SET stock = stock - 1 WHERE id = 1;
COMMIT;  -- lock released
```

**Optimistic locking**: Don't lock anything. Instead, check that the row
hasn't changed when you try to update it.

```sql
-- Read the current version
SELECT id, stock, version FROM products WHERE id = 1;
-- (stock=10, version=5)

-- Update only if version hasn't changed
UPDATE products
SET stock = 9, version = 6
WHERE id = 1 AND version = 5;

-- If 0 rows affected → someone else changed it → retry
```

**When to use which:**
- **Optimistic**: Low contention (conflicts are rare). Better throughput.
- **Pessimistic**: High contention (conflicts are frequent). Prevents wasted work.

---

### Q21: How do you handle database connections in a web application?

Use a **connection pool** to avoid the overhead of creating a new connection
for every request. A typical web app with 50 concurrent users does not need 50
database connections.

```python
# SQLAlchemy pool configuration
engine = create_engine(
    "postgresql://user:pass@localhost/mydb",
    pool_size=10,        # 10 persistent connections
    max_overflow=5,      # 5 extra if all 10 are busy
    pool_timeout=30,     # wait 30s for a connection before error
    pool_pre_ping=True,  # verify connections are alive
)
```

**External pooling with PgBouncer** for high-scale applications (thousands of
connections from multiple app instances, funneled through fewer DB connections).

---

### Q22: What is a materialized view and how does it differ from a regular view?

| | Regular View | Materialized View |
|---|-------------|------------------|
| Storage | No data stored (query runs every time) | Data is stored on disk |
| Speed | Same as the underlying query | Fast (reads precomputed data) |
| Freshness | Always up-to-date | Stale until refreshed |
| Use case | Simplify queries, access control | Cache expensive aggregations |

```sql
-- Regular view: executes the query every time you SELECT from it
CREATE VIEW v_stats AS SELECT department, AVG(salary) FROM employees GROUP BY department;

-- Materialized view: stores the result; must be refreshed manually
CREATE MATERIALIZED VIEW mv_stats AS SELECT department, AVG(salary) FROM employees GROUP BY department;
REFRESH MATERIALIZED VIEW mv_stats;
```

---

### Q23: Explain the `SELECT ... FOR UPDATE SKIP LOCKED` pattern.

This pattern is used to implement a **job queue** in PostgreSQL. Multiple
workers can pick up jobs without blocking each other.

```sql
-- Worker picks up one unprocessed job, skipping jobs locked by other workers
BEGIN;
SELECT * FROM jobs
WHERE status = 'pending'
ORDER BY created_at
LIMIT 1
FOR UPDATE SKIP LOCKED;

-- Process the job...

UPDATE jobs SET status = 'completed' WHERE id = <job_id>;
COMMIT;
```

This avoids the need for an external message queue for simple use cases.

---

### Q24: How does `pgvector` support AI/ML workloads?

`pgvector` adds vector data types and similarity search operators to
PostgreSQL, making it useful as a **vector database** for AI applications.

**Use cases:**
- Semantic search (find documents similar to a query embedding).
- Retrieval-Augmented Generation (RAG) for LLMs.
- Recommendation systems.
- Image similarity search.

**Distance functions:**
- `<->` L2 (Euclidean) distance
- `<=>` Cosine distance
- `<#>` Inner product (negative)

**Index types:**
- **IVFFlat**: Faster indexing, good recall with enough lists.
- **HNSW**: Better recall, slower to build, more memory.

```sql
-- Typical RAG pattern:
-- 1. Embed the user's question with an LLM
-- 2. Find the most relevant document chunks
SELECT content, embedding <=> $1 AS distance
FROM document_chunks
ORDER BY embedding <=> $1
LIMIT 5;
-- 3. Feed those chunks as context to the LLM for answering
```

---

### Q25: What is the difference between `EXISTS` and `IN`? Which is faster?

```sql
-- IN: fetches all values from the subquery, then checks membership
SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);

-- EXISTS: for each outer row, checks if the subquery returns any row
SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);
```

**Performance:**
- For **large subquery results**, `EXISTS` is often faster because it can
  short-circuit (stops as soon as one match is found).
- For **small subquery results**, `IN` can be faster (hash lookup).
- Modern PostgreSQL optimizes both similarly in many cases.
- **Beware of NULLs with NOT IN**: If the subquery returns any NULL, `NOT IN`
  returns no rows. `NOT EXISTS` handles NULLs correctly.

**Rule of thumb:** Use `EXISTS` / `NOT EXISTS` unless you have a specific
reason to use `IN`.

---

## Quick Reference Cheat Sheet

```
┌────────────────────────────────────────────────────────────────┐
│                    SQL QUICK REFERENCE                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Execution order: FROM → WHERE → GROUP BY → HAVING →           │
│                   SELECT → DISTINCT → ORDER BY → LIMIT         │
│                                                                │
│  JOIN types: INNER (both match), LEFT (all left + match),      │
│              RIGHT (all right + match), FULL (all from both),  │
│              CROSS (cartesian product)                          │
│                                                                │
│  Anti-join:  LEFT JOIN ... WHERE right.id IS NULL               │
│  Semi-join:  WHERE EXISTS (subquery)                           │
│                                                                │
│  Index rule: composite (A,B,C) serves A, AB, ABC only          │
│  Covering:   INCLUDE extra cols → index-only scan              │
│  Partial:    WHERE condition → smaller, faster index           │
│                                                                │
│  ACID: Atomicity, Consistency, Isolation, Durability           │
│  Default isolation: READ COMMITTED (PG), REPEATABLE READ (MY)  │
│                                                                │
│  N+1 fix:   JOIN or eager load or batch fetch                  │
│  Pagination: keyset (WHERE id > X) beats OFFSET for deep pages │
│  Deadlock:   always lock in consistent order                   │
│                                                                │
│  PG tools:  EXPLAIN ANALYZE, pg_stat_statements, VACUUM        │
│  PG types:  JSONB, arrays, tsvector, vector (pgvector)         │
│  PG locks:  FOR UPDATE, FOR UPDATE SKIP LOCKED, advisory locks │
│                                                                │
│  Python:    SQLAlchemy (Core/ORM), Alembic, asyncpg, PgBouncer │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```
