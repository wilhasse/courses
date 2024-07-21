import React from 'react';
import { range } from '../../utils';
import GuessSlot from '../GuessSlot';

function GuessList({ guesses }) {
  return (
    <div className="guess-results">
      {range(6).map((rowIndex) => (
        <GuessSlot key={rowIndex} value={guesses[rowIndex]} />
      ))}
    </div>
  );
}

export default GuessList;
