// test: https://svelte.dev/examples/hello-world
// Svelte 4 Store
//
<script>
	let x = 3
	let y
	$: {
		y = x + 10
	}
</script>
<h1>{y}</h1>
<button on:click={() => x++}> Increment </button>