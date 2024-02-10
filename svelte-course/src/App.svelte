<script>
  import Counter from './lib/Counter.svelte'
  import NobelPrize from './lib/NobelPrize.svelte'
  import NobelPrizeTable from './lib/NobelPrizeTable.svelte';

  let category = 'chemistry';
</script>

<main>
  <div class="card">
    <Counter />
  </div>
  <div class="card">
    <select bind:value={category}>
      <option>chemistry</option>
      <option>peace</option>
    </select>
    <NobelPrize {category} let:prizes>
      <NobelPrizeTable {prizes} />
    </NobelPrize>
    div>
</main>

<style>

</style>
