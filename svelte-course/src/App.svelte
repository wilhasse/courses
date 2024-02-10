<script>
  import Counter from './lib/Counter.svelte'
  import NobelPrize from './lib/NobelPrize.svelte'
  import NobelPrizeTable from './lib/NobelPrizeTable.svelte';
  import CategorySelect from './lib/CategorySelect.svelte';
  import AddPrizeModal from './lib/AddPrizeModal.svelte';
  import Button from './lib/Button.svelte';

  function handleAddPrize(event) {
		console.log(event.detail);
		showAddPrizeModal = false;
	}
   
  let category;
  let showAddPrizeModal = false;
</script>

<main>
  <div class="card">
    <Counter />
  </div>
  <div class="card">
    <CategorySelect bind:category/>
    <Button on:click={() => showAddPrizeModal = true}>Add new prize</Button>

    {#if showAddPrizeModal}
    <AddPrizeModal
      on:cancel={() => showAddPrizeModal = false}
		  on:submit={handleAddPrize}
    />
    {/if}
  
    <NobelPrize {category} let:prizes>
      <NobelPrizeTable {prizes} />
    </NobelPrize>
    div>
</main>

<style>

</style>
