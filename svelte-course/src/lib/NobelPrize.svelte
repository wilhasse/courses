<script>
import { fetchPrizes } from './prizes.js';
import category from './categoryStore';
</script>

<div>
{#await fetchPrizes($category) }
    Loading ...
{:then {prizes} } 
    <h1>Nobel Prize Winners</h1>
    <slot {prizes}></slot>
{:catch err}
    Error: {err}
{/await}
</div>

