import React from 'react';
import GuessInput from '../GuessInput';
import GuessList from '../GuessList';
import GuessSlot from '../GuessSlot';

import { sample } from '../../utils';
import { checkGuess } from '../../game-helpers';
import { WORDS } from '../../data';

// Pick a random word on every pageload.
const answer = sample(WORDS);
// To make debugging easier, we'll log the solution in the console.
console.info({ answer });

function AnswerOk( {attempts} ) {
  return (
    <div className="happy banner">
      <p>
        <strong>Congratulations!</strong> Got it in
        <strong>{attempts} guesses</strong>.
      </p>
    </div>
  );
}

function AnswerNotOk( {answer} ) {
  return (
    <div className="sad banner">
      <p>
        Sorry, the correct answer is <strong>{answer}</strong>.
      </p>
    </div>
  );
}

function Game() {
  const [guesses, setGuesses] = React.useState([]);
  const [guessesraw, setGuessesRaw] = React.useState([]);
  const [isDisabled, setIsDisabled] = React.useState(false);
  
  function handleSubmitGuess(guess) {
    const guessok = checkGuess(guess, answer);
    console.log(guessok);
    setGuesses([...guesses, guessok]);
    setGuessesRaw([...guessesraw, guess]);

    if(guesses.length === 6) {
      AnswerNotOk(answer);
    }
  }

  React.useEffect(() => {
    if (guessesraw.includes(answer)) {
      setIsDisabled(true);
    }
  }, [guessesraw]);

  function isAnswerOk(){

    if (guessesraw.includes(answer) && guessesraw.length > 0) {

      return true;
    } else {
      return false;
    }
  }

  return (
    <>
      <GuessList guesses={guesses} />
      <GuessInput handleSubmitGuess={handleSubmitGuess} isDisabled = {isDisabled} />
      {isAnswerOk() && <AnswerOk attempts={guessesraw.length} />}
      {guesses.length === 6 && <AnswerNotOk answer={answer} />}
    </>
  );
}

export default Game;
