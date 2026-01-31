module.exports = {
env: {
browser: true,
es2021: true,
node: true,
},
extends: [
'react-app',
'react-app/jest',
'eslint:recommended',
'plugin:@typescript-eslint/recommended',
'plugin:react/recommended',
'plugin:react-hooks/recommended',
'prettier',
],
parser: '@typescript-eslint/parser',
parserOptions: {
ecmaFeatures: {
jsx: true,
},
ecmaVersion: 'latest',
sourceType: 'module',
},
plugins: ['react', '@typescript-eslint', 'react-hooks', 'prettier'],
rules: {
'react/react-in-jsx-scope': 'off',
'react/prop-types': 'off',
'@typescript-eslint/explicit-function-return-type': 'off',
'@typescript-eslint/explicit-module-boundary-types': 'off',
'@typescript-eslint/no-explicit-any': 'warn',
'@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
'prettier/prettier': 'error',
},
settings: {
react: {
version: 'detect',
},
},
};
