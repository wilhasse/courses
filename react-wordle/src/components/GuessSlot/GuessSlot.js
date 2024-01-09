import React from 'react';
import { range } from '../../utils';

function GuessSlot({ value }) {
  return (
    <p className="guess">
      {range(5).map((cellIndex) => {
        return (
          <span key={cellIndex} className={`cell ${value && value[cellIndex] ? value[cellIndex].status : ''}`}>
            {value && value[cellIndex].letter ? value[cellIndex].letter : ''}
          </span>
        );
      })}
    </p>
  );
}

export default GuessSlot;
