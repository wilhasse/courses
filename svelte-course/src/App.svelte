<script>
  import Counter from './lib/Counter.svelte'
  import NobelPrize from './lib/NobelPrize.svelte'
  import NobelPrizeTable from './lib/NobelPrizeTable.svelte';
  import CategorySelect from './lib/CategorySelect.svelte';
  import Button from './lib/Button.svelte';

  let category = 'chemistry';
</script>

<main>
  <div class="card">
    <Counter />
  </div>
  <div class="card">
    <CategorySelect bind:category/>
    <Button on:click={() => alert('on Click')}>Add new prize</Button>
    <NobelPrize {category} let:prizes>
      <NobelPrizeTable {prizes} />
    </NobelPrize>
    div>
</main>

<style>

</style>
