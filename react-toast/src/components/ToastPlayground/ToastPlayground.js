import React, { useState} from 'react';

import Button from '../Button';
import ToastShelf from '../ToastShelf/ToastShelf';
import styles from './ToastPlayground.module.css';
import { ToastContext } from '../ToastProvider';

const VARIANT_OPTIONS = ['notice', 'warning', 'success', 'error'];

function ToastPlayground() {

  const [message, setMessage] = useState('');
  const [variant, setVariant] = useState('notice');
  const { createToast } = React.useContext(ToastContext);

  function handleCreateToast(event) {

    event.preventDefault();

    createToast(message, variant);

    setMessage('');
    setVariant(VARIANT_OPTIONS[0]);
  }

  return (
    <div className={styles.wrapper}>
      <header>
        <img alt="Cute toast mascot" src="/toast.png" />
        <h1>Toast Playground</h1>
      </header>

      <ToastShelf />

    <form className={styles.controlsWrapper} 
    onSubmit={event => {
      event.preventDefault();
      handleCreateToast(event);
      setMessage('');
      setVariant('notice');
      }}
    >

        <div className={styles.row}>
                <label
            htmlFor="message"
            className={styles.label}
            style={{ alignSelf: 'baseline' }}
        >
            Message
          </label>
          <div className={styles.inputWrapper}>
            <textarea id="message" className={styles.messageInput}             
            value={message}
            onChange={(event) => {
              setMessage(event.target.value);
            }}
 />
          </div>
        </div>

        <div className={styles.row}>
          <div className={styles.label}>Variant</div>
          <div
            className={`${styles.inputWrapper} ${styles.radioWrapper}`}
          >
          {VARIANT_OPTIONS.map((item) => {
            const id = `variant-${item}`;
            return (
            <label key={id} htmlFor={id}>
              <input                
                id="{id}"
                type="radio"
                name="variant"
                value={item}
                checked={item === variant}
                onChange={(event) => {
                setVariant(event.target.value);
              }}
              />
              {item}
            </label>
            );
          })}
          </div>
        </div>

        <div className={styles.row}>
          <div className={styles.label} />
          <div
            className={`${styles.inputWrapper} ${styles.radioWrapper}`}
          >
            <Button
             onClick={() => {
              setMessage(message);
              setVariant(variant);
             }}
            >Pop Toast!</Button>
          </div>
        </div>
      </form>
      </div>
  );
}

export default ToastPlayground;
