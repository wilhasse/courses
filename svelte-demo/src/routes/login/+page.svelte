<script lang="ts">
	import { onMount } from 'svelte';
	import { auth, user } from '$lib/firebase';
	import DatePicker from 'stwui/date-picker';
	//import { Datepicker, Input, initTE } from 'tw-elements';

	import { GoogleAuthProvider, signInWithPopup, signOut } from 'firebase/auth';
	import dayjs from 'dayjs';
	import 'dayjs/locale/pt-br';
	dayjs.locale('pt-br');
	const calendar = 'svg-path';

	// This function will be called when the captcha is verified
	onMount(async () => {
		// Dynamically import tw-elements
		const TE = await import('tw-elements');

		TE.initTE({ Datepicker: TE.Datepicker, Input: TE.Input });

		const datepickerBR = new TE.Datepicker(document.querySelector('#datepickerBR'), {
			title: 'Escolher data',
			monthsFull: [
				'Janeiro',
				'Fevereiro',
				'Março',
				'Abril',
				'Maio',
				'Junho',
				'Julho',
				'Agosto',
				'Setembro',
				'Outubro',
				'Novembro',
				'Dezembro'
			],
			monthsShort: [
				'Jan',
				'Fev',
				'Mar',
				'Abr',
				'Mai',
				'Jun',
				'Jul',
				'Ago',
				'Set',
				'Out',
				'Nov',
				'Dez'
			],
			weekdaysFull: [
				'Domingo',
				'Segunda-feira',
				'Terça-feira',
				'Quarta-feira',
				'Quinta-feira',
				'Sexta-feira',
				'Sábado'
			],
			weekdaysShort: ['Dom', 'Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb'],
			weekdaysNarrow: ['D', 'S', 'T', 'Q', 'Q', 'S', 'S'],
			okBtnText: 'Ok',
			clearBtnText: 'Limpar',
			cancelBtnText: 'Cancelar'
		});
	});

	async function signInWithGoogle() {
		const provider = new GoogleAuthProvider();
		const credential = await signInWithPopup(auth, provider);

		const idToken = await credential.user.getIdToken();

		const res = await fetch('/api/signin', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
				// 'CSRF-Token': csrfToken  // HANDLED by sveltekit automatically
			},
			body: JSON.stringify({ idToken })
		});
	}

	async function signOutSSR() {
		const res = await fetch('/api/signin', { method: 'DELETE' });
		await signOut(auth);
	}
</script>

<h2>Login</h2>

{#if $user}
	<h2 class="card-title">Welcome, {$user.displayName}</h2>
	<p class="text-center text-success">You are logged in</p>
	<button class="btn btn-warning" on:click={signOutSSR}>Sign out</button>
{:else}
	<button class="btn btn-primary" on:click={signInWithGoogle}>Sign in with Google</button>
{/if}

<DatePicker name="date" label="Date">
	<DatePicker.Label slot="label">Date</DatePicker.Label>
	<DatePicker.Trailing slot="trailing" data={calendar} />
</DatePicker>

<!-- TW Elements is free under AGPL, with commercial license required for specific uses. See more details: https://tw-elements.com/license/ and contact us for queries at tailwind@mdbootstrap.com -->
<div
	id="datepickerBR"
	class="relative mb-3 my-12"
	data-te-datepicker-init
	data-te-input-wrapper-init
>
	<input
		type="text"
		class="peer block min-h-[auto] w-full rounded border-0 bg-transparent px-3 py-[0.32rem] leading-[1.6] outline-none transition-all duration-200 ease-linear focus:placeholder:opacity-100 peer-focus:text-primary data-[te-input-state-active]:placeholder:opacity-100 motion-reduce:transition-none dark:text-neutral-200 dark:placeholder:text-neutral-200 dark:peer-focus:text-primary [&:not([data-te-input-placeholder-active])]:placeholder:opacity-0"
		placeholder="Select a date"
	/>
	<label
		for="floatingInput"
		class="pointer-events-none absolute left-3 top-0 mb-0 max-w-[90%] origin-[0_0] truncate pt-[0.37rem] leading-[1.6] text-neutral-500 transition-all duration-200 ease-out peer-focus:-translate-y-[0.9rem] peer-focus:scale-[0.8] peer-focus:text-primary peer-data-[te-input-state-active]:-translate-y-[0.9rem] peer-data-[te-input-state-active]:scale-[0.8] motion-reduce:transition-none dark:text-neutral-200 dark:peer-focus:text-primary"
		>Select a date</label
	>
</div>
