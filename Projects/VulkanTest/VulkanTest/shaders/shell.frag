#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in float fragAlpha;

layout(location = 0) out vec4 outColor;

layout(binding = 2) uniform sampler2D texSampler;

void main() 
{
   
	//outColor = texture(texSampler, fragTexCoord);		//Draw with textures 用贴图绘制
	outColor = vec4(texture(texSampler, fragTexCoord).xyz,fragAlpha);

	//outColor = vec4(fragColor, 0.1f);	//Draw colors for silhouette (FUR_LAYERS must be below 8 for the silhouette to show) 为轮廓绘制颜色(对于要展示的轮廓 FUR_LAYERS 必须小于8)
}