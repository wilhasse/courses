// test: https://svelte-5-preview.vercel.app
// Svelte 5 Runes
//
<script>
	let x = $state(5);
		let y = $derived(x+3)
</script>

<h1>{y}</h1>
<button onclick={() => x++}> Increment </button>