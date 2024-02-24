<script>
	import '../style.css';
  import {facts} from '../data.js';

  let fact = facts.shift();
  let explanation = "";
  let correct = 0;
  let completed = 0;

  function handleTrue() {
    
    // update counters
    correct++;
    completed++;

    // display explanation
    if (fact != undefined) {
      explanation = fact.explanation;
    }

    // next question
    fact = facts.shift();
  }

  function handleFalse() {

    // update counter
    completed++;

    // display explanation
    if (fact != undefined) {
     explanation = fact.explanation;
    }

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
      <button name="true" value="true" on:click={handleTrue}>true</button>
      <button name="false" value="false" on:click={handleFalse}>false</button>
    </div>
    {#if explanation}
    <div id="explanation">{explanation}</div>
    {/if}
    <button id="next-question" name="next-question">Next Question</button>
  </main>
