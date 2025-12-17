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

    vec2 tex_offset = 1.0 / textureSize(screenTexture, 0); // size of one texel

    int blurSizeX = 8;
    int blurSizeY = 8;
    vec4 total_hdr_bloom = vec4(0,0,0,0.0);
    for (float x=-blurSizeX; x<=blurSizeX; x++){
        for(float y=-blurSizeY; y<=blurSizeY;y++){
        vec4 samplePixel = texture(hdrTexture, TexCoords+vec2(x*tex_offset.x,y*tex_offset.y));
        samplePixel = samplePixel*10;
        float hdr_sample_luma = samplePixel.r*0.2126 + samplePixel.g*0.7152 + samplePixel.b*0.072;
        if (hdr_sample_luma > 9){

            total_hdr_bloom += (samplePixel/hdr_sample_luma);
        }
    }
    }
    total_hdr_bloom = total_hdr_bloom/(blurSizeX*blurSizeY);
    //total_hdr_bloom = total_hdr_bloom*0.1;

    worldColor += total_hdr_bloom*0.5;
    vec4 skyColor = texture(skyTexture, TexCoords);


    vec4 WorldAndSkyColor = (skyColor*(1-worldColor.a))+(worldColor*worldColor.a);
    if (worldColor.a == 0){
        worldColor = texture(skyTexture, TexCoords);
    }


    fragColor = WorldAndSkyColor;

    //fragColor = total_hdr_bloom;



}
