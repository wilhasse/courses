<script>
	let user = '';
	let password = '';
</script>

<div class="login-container">
	<form method="POST" class="login-form">
		<div class="form-group">
			<label for="user">Usuário</label>
			<input id="user" name="user" type="user" bind:value={user} required />
		</div>
		<div class="form-group">
			<label for="password">Senha</label>
			<input id="password" name="password" type="password" bind:value={password} required />
		</div>
		<button type="submit" class="login-button">Login</button>
	</form>
</div>

<style>
	.login-container {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100vh;
		background-color: #f0f4f8;
	}

	.login-form {
		padding: 2rem;
		background-color: #fff;
		border-radius: 0.5rem;
		box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.1);
		width: 300px;
	}

	.form-group {
		margin-bottom: 1rem;
		display: flex;
		flex-direction: column;
	}

	.form-group input {
		padding: 0.5rem;
		font-size: 1rem;
		border: 2px solid #ddd;
		border-radius: 0.25rem;
	}

	.form-group label {
		margin-bottom: 0.5rem;
		font-size: 1rem;
	}

	.login-button {
		padding: 0.5rem;
		width: 100%;
		font-size: 1rem;
		color: white;
		background-color: #007bff;
		border: none;
		border-radius: 0.25rem;
		cursor: pointer;
	}

	.login-button:hover {
		background-color: #0056b3;
	}
</style>
