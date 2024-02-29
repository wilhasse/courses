<script lang="ts">
	export let emoji: string;
	export let selected: boolean;
</script>

<div class="square" class:flipped={selected}>
<span>{emoji}</span>
<button on:click></button>
</div>

<style>
	.square {
		display: flex;
		justify-content: center;
		align-items: center;
		background-color: yellow;
	}

	.flipped {

	}
	
	button {
		font-size: 6em;
		border: none;
		background: #eee;
	}
</style>