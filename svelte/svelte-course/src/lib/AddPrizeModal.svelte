<script>
	import { createEventDispatcher } from 'svelte';
	import CategorySelect from './CategorySelect.svelte';

	const dispatch = createEventDispatcher();
	
	let year;
	let category;
	let firstname;
	let surname;
	let motivation;
	
	function handleCancel(event) {
		dispatch('cancel');
	}
	
	function handleSubmit(event) {
		dispatch('submit', {
			year, category, laureates: [ {firstname, surname, motivation} ]
		})
	}
</script>

<style>
	form {
		position: fixed;
		max-height: 70vh;
		overflow: auto;
		background: white;
		margin: 0 auto;
		box-shadow: 0 0 5px rgba(0,0,0,0.5);
		padding: 10px;
	}
	
	input {
		display: block;
	}
</style>

<form on:submit|preventDefault={handleSubmit}>
	<p>
		<label>
			Year:
			<input name="year" bind:value={year} required />
		</label>
	</p>
	<p>
		<label for="category">
			Category:
			<CategorySelect bind:category />
		</label>
	</p>
	<p>
		<label>
			Laureate First Name:
			<input name="firstname" bind:value={firstname} required />
		</label>
	</p>
	<p>
		<label>
			Laureate Last Name:
			<input name="surname" bind:value={surname} required />
		</label>
	</p>
	<p>
		<label>
			Motivation:
			<textarea name="motivation" bind:value={motivation} required/>
		</label>
	</p>
	<p>
		<input type="submit" value="Add prize"/>
		<input type="button" value="Cancel" on:click={handleCancel}/>
	</p>
</form>