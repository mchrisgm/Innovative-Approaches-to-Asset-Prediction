# Trading Strategy

## The signal
The approach for opening and closing positions relies solely on the CNN model output. After the prediction function is given the required data with the correct lookback period, the model then gives a forecast between the range of -1 to 1.

## The algorithm
The algorithm is responsible for converting the -1 to 1 signal into a `long` position, `short` position, `holding` the previous position or `closing` all the positions.
This happens using:
```python
if prediction > 0:
    if current_position <= 0:
        current_position = prediction
        return prediction   # Go long
elif prediction < 0:
    if current_position >= 0:
        current_position = prediction
        return prediction   # Go short
else:  # prediction must be 0
    if current_position != 0:
        current_position = 0
        return 0.0  # Close position
```
Where:
- **prediction**: the CNN model output
- **current_position**: The position the model holds at any point in time
- **return prediction**: The final decision made by the strategy for the specific point in time
