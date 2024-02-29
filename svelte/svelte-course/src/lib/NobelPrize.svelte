<script>
import prizes from './selectedPrizesStore';
</script>

<div>
{#if $prizes === null }
    Loading ...
{:else if $prizes.error}
    {$prizes.error}
{:else}
    <slot prizes={$prizes} />
{/if}
</div>

