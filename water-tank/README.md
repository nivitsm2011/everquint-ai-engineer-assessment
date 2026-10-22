# Water Tank Calculator

## 1. Project Overview

The Water Tank Calculator solves the Trapping Rain Water problem. Given an array of non-negative integers representing building heights, the application calculates how much water can be trapped between the buildings.

The application provides a simple web interface where users can enter building heights, calculate trapped water, and view a visualization of the buildings and trapped water.

## 2. Technologies Used

- HTML5 — webpage structure
- CSS3 — styling and responsive layout
- JavaScript — calculation and input validation
- SVG — visualization of buildings and trapped water

## 3. Algorithm

The water calculation uses the two-pointer technique.

Two pointers start at opposite ends of the height array. The algorithm maintains the maximum height encountered from each side and processes the side with the smaller current boundary.

For each processed position, it either updates the maximum height or adds the difference between the maximum height and the current building height to the total trapped water.

### Complexity

- Time complexity: O(n)
- Auxiliary space complexity: O(1) for the calculation algorithm

Here, n is the number of building heights.

The visualization uses additional arrays to calculate the left and right maximum heights for drawing the water accurately.

## 4. Features

- Accepts comma-separated building heights
- Calculates trapped water using the two-pointer algorithm
- Validates empty and invalid input
- Rejects negative heights and non-integer values
- Displays the total trapped water
- Visualizes buildings and trapped water using SVG
- Supports up to 100 building heights

## 5. How to Run

1. Download or clone the repository.
2. Open the `water-tank` folder.
3. Open `index.html` in a modern web browser.
4. Enter building heights separated by commas.
5. Click **Calculate Water** to view the result and visualization.

No additional packages are required.

## 6. Test Cases

| Input                 |        Expected Output |
| --------------------- | ---------------------: |
| `3,0,2,0,4`           |                7 units |
| `0,4,0,0,0,6,0,6,4,0` |               18 units |
| `4,2,0,3,2,5`         |                9 units |
| `1,2,3,4`             |                0 units |
| `3,3,3,3`             |                0 units |
| `3,abc,4`             | Input validation error |
| `-2,3,4`              | Input validation error |
| Empty input           | Input validation error |

The expected results above were checked during development.

## 7. Conclusion

This project demonstrates the use of the two-pointer technique to solve the Trapping Rain Water problem efficiently. It combines algorithm implementation, input validation, and SVG visualization in a browser-based application.
