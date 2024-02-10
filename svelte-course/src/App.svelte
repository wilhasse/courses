<script>
  import Counter from './lib/Counter.svelte'
  import NobelPrize from './lib/NobelPrize.svelte'
  import NobelPrizeTable from './lib/NobelPrizeTable.svelte';
  import CategorySelect from './lib/CategorySelect.svelte';

  let category = 'chemistry';
</script>

<main>
  <div class="card">
    <Counter />
  </div>
  <div class="card">
    <CategorySelect bind:category/>
    <NobelPrize {category} let:prizes>
      <NobelPrizeTable {prizes} />
    </NobelPrize>
    div>
</main>

<style>

</style>
