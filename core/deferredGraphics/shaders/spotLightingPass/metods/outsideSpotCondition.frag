bool outsideSpotCondition(vec3 coordinates, float type)
{
    if(type==0.0f){
        return sqrt(coordinates.x*coordinates.x + coordinates.y*coordinates.y) >= coordinates.z;
    }else{
        return abs(coordinates.x) >= abs(coordinates.z) || abs(coordinates.y) >= abs(coordinates.z);
    }
}
