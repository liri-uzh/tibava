const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { spawnSync } = require('node:child_process');

// Start fresh processes so each timezone is explicit before Date/Intl initialize.
if (!process.env.HISTORY_TIMEZONE_TEST) {
  for (const timezone of ['Europe/Zurich', 'UTC', 'America/New_York']) {
    const result = spawnSync(process.execPath, [__filename], {
      env: { ...process.env, TZ: timezone, HISTORY_TIMEZONE_TEST: '1' },
      stdio: 'inherit',
    });
    assert.equal(result.status, 0, timezone);
  }
  process.exit(0);
}

const source = fs.readFileSync(path.join(__dirname, '../src/components/History.vue'), 'utf8');
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace(/^import .*;$/gm, '')
  .replace('export default', 'module.exports =');
const context = {
  module: { exports: {} }, mapStores: () => ({}),
  usePlayerStore: {}, usePluginRunStore: {}, usePluginRunResultStore: {}, ModalPlugin: {},
};
vm.runInNewContext(script, context);
const history = context.module.exports;
const instance = { ...history.methods, pluginName: type => type };
const display = value => instance.formatDate(instance.dateTimestamp(value));
const expected = {
  'Europe/Zurich': ['2026-09-17 08:58:41', '2026-01-17 07:58:41'],
  UTC: ['2026-09-17 06:58:41', '2026-01-17 06:58:41'],
  'America/New_York': ['2026-09-17 02:58:41', '2026-01-17 01:58:41'],
}[process.env.TZ];
assert.equal(display('2026-09-17T06:58:41.390Z'), expected[0]);
assert.equal(display('2026-01-17T06:58:41.390Z'), expected[1]);
for (const value of [undefined, null, '', 'bad-date']) assert.equal(display(value), '');
assert.equal(instance.formatDate(NaN), '');

// The autumn clock change reverses local clock order, but not chronological order.
const dates = ['2026-10-25T00:45:00Z', '2026-10-25T01:15:00Z',
  '2026-09-17T06:58:41.390Z', '2026-09-17T06:58:41.391Z'];
instance.playerStore = { videoId: 42 };
instance.pluginRunStore = { forVideo: id => {
  assert.equal(id, 42);
  return [...dates, null, undefined, 'bad-date'].map((date, index) => ({
    date, type: String(index), progress: '0.5', status: 'DONE',
  }));
} };
const rows = history.computed.pluginRuns.call(instance);
assert.deepEqual(Array.from(rows, row => row.type), ['1', '0', '3', '2', '4', '5', '6']);
assert.ok(rows.slice(0, 4).every(row => typeof row.date === 'number'));
assert.ok(rows.slice(4).every(row => row.date === null));
// User-triggered table sorting also receives numeric values, not display strings.
assert.deepEqual(Array.from(rows.slice(0, 4).sort((a, b) => a.date - b.date), row => row.type),
  ['2', '3', '0', '1']);
if (process.env.TZ === 'Europe/Zurich') {
  assert.equal(instance.formatDate(rows[0].date), '2026-10-25 02:15:00');
  assert.equal(instance.formatDate(rows[1].date), '2026-10-25 02:45:00');
}
const template = source.split('<script>')[0].replace(/^<template>\s*/, '').replace(/<\/template>\s*$/, '');
const compiled = require('vue-template-compiler').compile(template);
assert.deepEqual(compiled.errors, []);
const Vue = require('vue');
const render = new Function(compiled.render);
const component = new Vue({
  data: () => ({ ...history.data(), pluginRuns: rows, numRunningPlugins: 0 }),
  methods: { ...history.methods, $t: value => value },
  render, staticRenderFns: compiled.staticRenderFns.map(fn => new Function(fn)),
});
const dialog = component._render();
const table = dialog.children.find(child => child.tag === 'v-data-table');
for (const row of rows) {
  const cell = table.data.scopedSlots['item.date']({ value: row.date });
  assert.equal(cell[0].text.trim(), instance.formatDate(row.date));
}
component.$destroy();
console.log(`History formatting, rendering, DST, invalid dates, and sorting passed (${process.env.TZ}).`);
