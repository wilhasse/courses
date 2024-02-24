<script>
	import '../style.css';
	import { facts } from '../data.js';

	let fact = facts.shift();
	let explanation = '';
	let correct = 0;
	let completed = 0;
	let buttonTrue = '';
	let buttonFalse = '';
  let nextButton = 'Next Question';
	let enableNext = false;
  let enableAnswer = true;

	// @ts-ignore
	function handleAnswer(event) {
		const guess = event.target.value;

    // @ts-ignore
    if (guess === fact.answer) {
			// correct answer! Adding class style
			if (event.target.name === 'true') {
				buttonTrue = 'correct';
			} else {
				buttonFalse = 'correct';
			}
      correct += 1;

		} else {
			// wrong answer! Adding class style
			if (event.target.name === 'true') {
				buttonTrue = 'incorrect';
			} else {
				buttonFalse = 'incorrect';
			}
		}

		// update counter
		completed++;

		// display explanation
		if (fact != undefined) {
			explanation = fact.explanation;
		}

		// last one ?
		if (facts.length === 0) {
			enableNext = false;
      nextButton = "No more questions!";
      enableAnswer = false;
		} else {
			enableNext = true;
		}
	}

	function nextQuestion() {
		// clear text and classes
		explanation = '';
		buttonTrue = '';
		buttonFalse = '';

		// next question
		fact = facts.shift();
	}
</script>

<header>
	<h1>Quiz.js</h1>
	<p>Do you know JS? Find out!</p>
	<div id="score">
		Score: <span id="correct">{correct}</span> / <span id="completed">{completed}</span>
	</div>
</header>
<main>
	<div id="statement">{fact?.statement}</div>
	<div id="options">
		<button disabled={!enableAnswer} class={buttonTrue} name="true" value="true" on:click={handleAnswer}>true</button>
		<button disabled={!enableAnswer} class={buttonFalse} name="false" value="false" on:click={handleAnswer}>false</button>
	</div>
	{#if explanation}
		<div id="explanation">{explanation}</div>
	{/if}
	<button disabled={!enableNext} id="next-question" name="next-question" on:click={nextQuestion}
		>{nextButton}</button
	>
</main>
