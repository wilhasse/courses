<script>
export let onClick;

function handleClick() {

    onClick();
}
</script>

<style>
	button {
		background: blue;
		color: white;
		border-radius: 0.5em;
		cursor: pointer;
	}
</style>

<button on:click={handleClick}>
	<slot />
</button>