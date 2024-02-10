<script>
  import App from '../App.svelte';


import { fetchPrizes , fetchPrizesByCategory } from './prizes.js';
export let category;

function fetchData(category) {

    return  category ? fetchPrizesByCategory(category) : fetchPrizes();
}

</script>

<div>
{#await fetchData(category) }
    Loading ...
{:then {prizes} } 
    <h1>Nobel Prize Winners</h1>
    <slot {prizes}></slot>
{:catch err}
    Error: {err}
{/await}
</div>

