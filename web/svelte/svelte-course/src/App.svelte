<script>
  import Counter from './lib/Counter.svelte'
  import NobelPrize from './lib/NobelPrize.svelte'
  import NobelPrizeTable from './lib/NobelPrizeTable.svelte';
  import CategorySelect from './lib/CategorySelect.svelte';
  import AddPrizeModal from './lib/AddPrizeModal.svelte';
  import Button from './lib/Button.svelte';
  import date from './lib/dateStore.js';
  import category from './lib/categoryStore';
  import { addPrize } from './lib/prizesStore.js';

  function handleAddPrize(event) {

		console.log(event.detail);
    addPrize(event.detail);
		showAddPrizeModal = false;
	}

  let showAddPrizeModal = false;
</script>

<main>
  <div class="card">
    <Counter />
  </div>
  <div class="card">
    <p> The current time is {$date.toLocaleTimeString()}</p>
    <CategorySelect bind:category={$category} />
    <Button on:click={() => showAddPrizeModal = true}>Add new prize</Button>

    {#if showAddPrizeModal}
    <AddPrizeModal
      on:cancel={() => showAddPrizeModal = false}
		  on:submit={handleAddPrize}
    />
    {/if}
  
    <NobelPrize let:prizes>
      <NobelPrizeTable {prizes} />
    </NobelPrize>
    div>
</main>

<style>

</style>
