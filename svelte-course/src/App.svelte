<script>
  import Counter from './lib/Counter.svelte'
  import NobelPrize from './lib/NobelPrize.svelte'
  import NobelPrizeTable from './lib/NobelPrizeTable.svelte';
</script>

<main>
  <div class="card">
    <Counter />
  </div>
  <div class="card">
    <NobelPrize let:prizes>
      <NobelPrizeTable {prizes} />
    </NobelPrize>
    div>
</main>

<style>

</style>
