<template>
  <main class="container">
    <h1>NumJS Vue + Vite 示例</h1>
    <p v-if="ready">当前后端：<strong>{{ backend }}</strong></p>
    <p v-else>后端初始化中...</p>

    <section v-if="ready" class="grid">
      <article class="panel">
        <h2>矩阵加法</h2>
        <pre>A = {{ format(aArray) }}</pre>
        <pre>B = {{ format(bArray) }}</pre>
        <pre>A + B = {{ format(sumArray) }}</pre>
      </article>

      <article class="panel">
        <h2>Clip &amp; Where</h2>
        <pre>Clip(A, 0, 3) = {{ format(clippedArray) }}</pre>
        <pre>Where(mask, A, B) = {{ format(whereArray) }}</pre>
      </article>
    </section>
  </main>
</template>

<script setup lang="ts">
import {
  Matrix,
  add,
  clip,
  init,
  where,
  backendKind,
} from "@jayce789/numjs";
import { onMounted, ref } from "vue";

const ready = ref(false);
const backend = ref<string>("unknown");

const a = new Matrix([1, 2, 3, 4], 2, 2);
const b = new Matrix([5, 6, 7, 8], 2, 2);
const mask = new Matrix([0, 1, 1, 0], 2, 2);

const aArray = ref<number[]>([]);
const bArray = ref<number[]>([]);
const sumArray = ref<number[]>([]);
const clippedArray = ref<number[]>([]);
const whereArray = ref<number[]>([]);

onMounted(async () => {
  await init();
  backend.value = backendKind();
  ready.value = true;

  aArray.value = Array.from(a.toArray());
  bArray.value = Array.from(b.toArray());
  sumArray.value = Array.from(add(a, b).toArray());
  clippedArray.value = Array.from(clip(a, 0, 3).toArray());
  whereArray.value = Array.from(where(mask, a, b).toArray());
});

function format(values: number[]) {
  return `[${values.map((v) => v.toFixed(2)).join(", ")}]`;
}
</script>

<style scoped>
.container {
  font-family: system-ui, sans-serif;
  margin: 2rem auto;
  padding: 0 1rem;
  max-width: 720px;
}

.grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  margin-top: 1.5rem;
}

.panel {
  background: #f5f5f5;
  border-radius: 8px;
  padding: 1rem;
}

pre {
  background: #1e1e1e;
  color: #f5f5f5;
  padding: 0.5rem 0.75rem;
  border-radius: 4px;
  overflow-x: auto;
}
</style>
