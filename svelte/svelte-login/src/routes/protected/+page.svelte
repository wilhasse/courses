<script>
    import { goto } from '$app/navigation';
	export let data;

    async function handleSignOut() {
        await goto(`/?signout=${data.user.user}`);
    }    
</script>

<h1>Login {data.user.user} OK!</h1>

<button type="submit" on:click={handleSignOut}>Logout</button>
