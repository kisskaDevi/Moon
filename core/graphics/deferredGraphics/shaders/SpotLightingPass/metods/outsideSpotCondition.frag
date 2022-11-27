bool outsideSpotCondition(vec3 coordinates, float type)
{
    if(type==0.0f){
        return coordinates.x*coordinates.x + coordinates.y*coordinates.y - coordinates.z*coordinates.z > 0.0f;
    }else{
        return abs(coordinates.x) > 1.0f || abs(coordinates.y) > 1.0f || abs(coordinates.z) > 1.0f;
    }
}
