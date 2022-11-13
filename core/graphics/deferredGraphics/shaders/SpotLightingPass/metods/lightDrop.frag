float lightDrop(float distance)
{
    float C = 1.0f;
    float L = 0.0866f*exp(-0.00144f*distance);
    float Q = 0.0283f*exp(-0.00289f*distance);

    return C + L * distance + Q * distance * distance;
}