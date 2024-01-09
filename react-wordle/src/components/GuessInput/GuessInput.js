import React from 'react';

function GuessInput({ handleSubmitGuess,isDisabled }) {

  const [guess2, setGuess2] = React.useState('');

  const handleInputChange = (e) => {
    const upperCasedValue = e.target.value.toUpperCase();
    setGuess2(upperCasedValue);
  };

  return (
    <form
      className="guess-input-wrapper"
      onSubmit={(event) => {
        event.preventDefault();
        console.log({guess2});
        handleSubmitGuess(guess2);
        setGuess2('');
      }}
    >
      <label htmlFor="guess-input">Enter guess:</label>
      <input
        required
        minLength="5"
        maxLength="5"
        id="guess-input"
        type="text"        
        value={guess2}
        disabled={isDisabled}
        onChange={handleInputChange}
      />
    </form>
  );
}

export default GuessInput;
