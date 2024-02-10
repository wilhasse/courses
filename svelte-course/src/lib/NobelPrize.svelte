<script>
import { fetchPrizes } from './prizes.js';
export let category;
</script>

<div>
{#await fetchPrizes(category) }
    Loading ...
{:then {prizes} } 
    <h1>Nobel Prize Winners</h1>
    <slot {prizes}></slot>
{:catch err}
    Error: {err}
{/await}
</div>

