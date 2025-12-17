#version 330 core
in vec2 TexCoords;
out vec4 fragColor;
uniform sampler2D screenTexture;
uniform sampler2D hdrTexture;
//uniform sampler2D uiTexture;
uniform sampler2D skyTexture;
uniform sampler2D pauseTexture;

void main()
{
    fragColor = (texture(screenTexture, TexCoords));
    if (TexCoords.x > 0.5){
    //fragColor = (texture(uiTexture, TexCoords));
    //fragColor = vec4(0,1.0,0,1.0);
    }
    //fragColor = vec4(gl_FragCoord.xy*vec2(400,300),0.0,1.0);
    


    //float UI_alpha =  (texture(uiTexture, TexCoords)).a;

    vec4 worldColor = texture(screenTexture, TexCoords);

    vec4 hdrColor = texture(hdrTexture, TexCoords);
    hdrColor = hdrColor*10;
    //worldColor = worldColor*10;

    float hdr_luma = hdrColor.r*0.2126 + hdrColor.g*0.7152 + hdrColor.b*0.072;
    //vec4 hdrColor = vec4(0,0,0,0);
    if (hdr_luma > 1){
        //hdrColor = worldColor;
    }else{
        hdrColor = vec4(0,0,0,0);
    }
    //worldColor = hdrColor;



    vec4 skyColor = texture(skyTexture, TexCoords);


    vec4 WorldAndSkyColor = (skyColor*(1-worldColor.a))+(worldColor*worldColor.a);
    if (worldColor.a == 0){
        worldColor = texture(skyTexture, TexCoords);
        //worldColor.g = 1;
        //worldColor = skyTexture;
    }
    //worldColor = texture(skyTexture, TexCoords);
    //fragColor = (WorldAndSkyColor*(1-UI_alpha))+ (texture(uiTexture, TexCoords));

    fragColor = WorldAndSkyColor;
    //fragColor = (fragColor*(1-texture(pauseTexture, TexCoords).a))+texture(pauseTexture, TexCoords);


    //fragColor = texture(skyTexture, TexCoords);
    //fragColor = texture(skyTexture, TexCoords);
    //fragColor = vec4(UI_alpha,0,0,1);
    //fragColor = worldColor;



}
