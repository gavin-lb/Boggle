# Boggle

Script that plays the Android app "Boggle With Friends"; a word game about making words on a 4x4 scrabble-like gameboard, with scrabble-like scoring.

Requires Android Studio to be installed as it uses ADB (Android Debug Bridge) to interface with the device, which can either be an emulated Android device or a physical device connected via USB. 

The script pulls a screenshot from the device, uses a custom OCR algorithm (implimented with PIL) to read the tiles and multipliers from the gameboard, constructs a graph representing the gameboard then uses a DFS to find all valid words (an important optimization here being that branches which are not prefixes of any allowed word are pruned; a [Directed acyclic word graph (DAWG)](https://en.wikipedia.org/wiki/Deterministic_acyclic_finite_state_automaton) is used to efficiently check for prefixes), scores all found words, sorts them according to most-points-per-input, then finally uses the Android Studio tool [Monkeyrunner](https://developer.android.com/studio/test/monkeyrunner) to input the words into the device. 

Here is an example of it working in action: 

![Boggle](https://github.com/gavin-lb/gavin-lb/blob/main/boggle.gif)
