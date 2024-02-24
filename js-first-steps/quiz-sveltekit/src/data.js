const facts = [
    {
      "statement": "JavaScript was invented in 1995",
      "answer": "true",
      "explanation": "Brendan Eich created JS at Netscape in 1995. The initial version of the language was written in just 10 days."
    },
    {
      "statement": "Strings in JS are editable values",
      "answer": "false",
      "explanation": "In JavaScript strings are immutable values, meaning they cannot be edited; however, they can replaced with new, different strings."
    },
    {
      "statement": "1 + 1 === 2",
      "answer": "true",
      "explanation": "The plus operator gives the sum of two numbers."
    },
    {
      "statement": "'1' + '1' === '2'",
      "answer": "false",
      "explanation": "The plus operator concatenates (joins together) strings, so '1' + '1' === '11'."
    },
    {
      "statement": "typeof ['J', 'S'] === 'array'",
      "answer": "false",
      "explanation": "Arrays have the type 'object'. In JS, everything is either a primitive data type (e.g. 'string', 'number') or an object. Arrays are a kind of object with some special properties.  "
    }
  ];

export { facts };
