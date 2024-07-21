'use client';
import React from 'react';
import clsx from 'clsx';
import {
  Play,
  Pause,
  RotateCcw,
} from 'react-feather';

import Card from '@/components/Card';
import VisuallyHidden from '@/components/VisuallyHidden';

import styles from './CircularColorsDemo.module.css';

const COLORS = [
  { label: 'red', value: 'hsl(348deg 100% 60%)' },
  { label: 'yellow', value: 'hsl(50deg 100% 55%)' },
  { label: 'blue', value: 'hsl(235deg 100% 65%)' },
];

function CircularColorsDemo() {

  const [timeElapsed, setTimeElapsed] = React.useState(0);
  const [selectedColor, setSelectedColor] = React.useState(COLORS[0]);
  const [isPlaying, setIsPlaying] = React.useState(false);
  
  React.useEffect(() => {
    if (!isPlaying) {
      return;
    }
  
    const intervalId = window.setInterval(() => {
      // Update timeElapsed using an updater function to avoid stale state
      setTimeElapsed(prevTimeElapsed => {
        const newTimeElapsed = prevTimeElapsed + 1;
        // Update color based on new timeElapsed value
        setSelectedColor(COLORS[newTimeElapsed % COLORS.length]);
        return newTimeElapsed;
      });
    }, 1000);
  
    return () => {
      window.clearInterval(intervalId);
    };
  }, [isPlaying]); // Only recreate the interval when isPlaying changes
  
  const handleClick = () => {
    setIsPlaying(!isPlaying);
  };

  const handleResetClick = (message) => {
    console.log(message);
    setTimeElapsed(0);
  };

  return (
    <Card as="section" className={styles.wrapper}>
      <ul className={styles.colorsWrapper}>
        {COLORS.map((color, index) => {
          const isSelected =
            color.value === selectedColor.value;

          return (
            <li
              className={styles.color}
              key={index}
            >
              {isSelected && (
                <div
                  className={
                    styles.selectedColorOutline
                  }
                />
              )}
              <div
                className={clsx(
                  styles.colorBox,
                  isSelected &&
                    styles.selectedColorBox
                )}
                style={{
                  backgroundColor: color.value,
                }}
              >
                <VisuallyHidden>
                  {color.label}
                </VisuallyHidden>
              </div>
            </li>
          );
        })}
      </ul>

      <div className={styles.timeWrapper}>
        <dl className={styles.timeDisplay}>
          <dt>Time Elapsed</dt>
          <dd>{timeElapsed}</dd>
        </dl>
        <div className={styles.actions}>
          <button onClick={handleClick}>
          {isPlaying ? <Pause /> : <Play/>}

            <VisuallyHidden>Play</VisuallyHidden>
          </button>
          <button onClick={handleResetClick}>
            <RotateCcw />
            <VisuallyHidden>Reset</VisuallyHidden>
          </button>
        </div>
      </div>
    </Card>
  );
}

export default CircularColorsDemo;
