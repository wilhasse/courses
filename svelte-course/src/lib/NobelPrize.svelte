<script>
import prizes from './prizesStore.js';
import category from './categoryStore';
</script>

<div>
{#if $prizes === null }
    Loading ...
{:else if $prizes.error}
    {$prizes.error}
{:else}
    <slot prizes={$prizes.filter(prize => prize.category === $category)} />
{/if}
</div>

