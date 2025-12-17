#version 330 core
in vec2 TexCoords;
out vec4 fragColor;
uniform sampler2D screenTexture;
uniform sampler2D hdrTexture;
uniform sampler2D skyTexture;
uniform sampler2D pauseTexture;

void main()
{
    fragColor = (texture(screenTexture, TexCoords));
    vec4 worldColor = texture(screenTexture, TexCoords);

    vec4 hdrColor = texture(hdrTexture, TexCoords);
    hdrColor = hdrColor*10;

    float hdr_luma = hdrColor.r*0.2126 + hdrColor.g*0.7152 + hdrColor.b*0.072;
    //vec4 hdrColor = vec4(0,0,0,0);
    if (hdr_luma > 1){

    }else{
        hdrColor = vec4(0,0,0,0);
    }
    vec4 skyColor = texture(skyTexture, TexCoords);


    vec4 WorldAndSkyColor = (skyColor*(1-worldColor.a))+(worldColor*worldColor.a);
    if (worldColor.a == 0){
        worldColor = texture(skyTexture, TexCoords);
    }

    fragColor = WorldAndSkyColor;



}
