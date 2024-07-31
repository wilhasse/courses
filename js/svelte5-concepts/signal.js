// test: https://codepen.io/
let currentEffect

function createSignal(value) {
const rx = new Set()
  return {
    getValue: () => {
      if (currentEffect) rx.add(currentEffect)        
      return value
    },
    setValue: (newVal) => {
      value = newVal
      for (const reaction of rx) {
        let oldEffect = currentEffect
        currentEffect = null
        reaction()
        currentEffect = oldEffect
      }
    }
  }
}

function effect(fn) {
  currentEffect = fn
  fn()
  currentEffect = undefined
}

const age = createSignal(31) 
const message = createSignal("you're not young anymore!")
effect( () => {
  console.log("Effect being re-run!!")
  console.log(age.getValue())
  if (age.getValue() >35) {
    console.log(message.getValue())
  }
})

age.setValue(40)
message.setValue("you're REALLy over 35!")