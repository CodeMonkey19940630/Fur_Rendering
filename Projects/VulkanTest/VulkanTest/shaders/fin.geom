#version 450
#define FUR_LENGTH 0.01

layout(set = 0,binding = 0) uniform UniformBufferObject 
{
    mat4 model;
    mat4 view;
    mat4 proj;
	
} ubo;

layout(set = 0, binding = 1) uniform EyeUniformBufferObject 
{
  	vec3 pos;
	
} eye;



in gl_PerVertex
{
    vec4  gl_Position;
  
} gl_in[];

//输入

layout(triangles_adjacency) in;

layout(location = 0) in vec3 inColor[6];
layout(location = 1) in vec3 inNormal[6];
layout(location = 2) in vec2 inTexCoord[6];

//输出

layout(triangle_strip, max_vertices = 30) out;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out float fragAlpha;

//计算法线
vec3 getNormal(vec3 A, vec3 B, vec3 C)
{
	return normalize(cross(C-A,B-A));
}

void main() 
{


	
	//Current triangle world positions
	// 当前三角形的世界位置
	vec4 wp1 = ubo.model*gl_in[0].gl_Position;
	vec4 wp2 = ubo.model*gl_in[2].gl_Position;
	vec4 wp3 = ubo.model*gl_in[4].gl_Position;
	
	vec3 N = getNormal(wp1.xyz,wp2.xyz,wp3.xyz);			//当前三角形法线
	vec3 basePoint = ((wp1+wp2+wp3)/3.0f).xyz;				//当前三角形重心
	vec3 viewDirection = normalize(basePoint - eye.pos.xyz);	//相机到重心的向量

	float dotView = dot(N, viewDirection);					

	if(dotView < 0)			//当前三角形是面向相机
	{


	    for(int i =0; i < gl_in.length(); i+=2)		//For each of the adjacent triangles 对于每个相邻的三角形
        {
		   int prevVer = (i+2)%6;					//The previous vertex using modulo 使用模的上一个顶点
		   

		   //临近三角形的位置
			wp1 = ubo.model*gl_in[i].gl_Position;
			wp2 = ubo.model*gl_in[i+1].gl_Position;
			wp3 = ubo.model*gl_in[prevVer].gl_Position;

			//Similarly compute the dot view
			vec3 N = getNormal(wp1.xyz,wp2.xyz,wp3.xyz);
			vec3 basePoint = ((wp1+wp2+wp3)/3.0f).xyz;
			vec3 viewDirection = normalize(basePoint - eye.pos.xyz);
			viewDirection = normalize(viewDirection);

			
			float dotView = dot(N, viewDirection);

			if(dotView >= 0)		// 临近的三角形是背向相机
			{



     	        fragAlpha = 1;
		        fragColor = vec3(1.0,0.0,0.0);	//Red silhouettte //红色轮廓
		        fragTexCoord = vec2(0.0f,1.0f);	//Because of lack of good fin texture use the same fur one	 由于缺乏良好的鳍状纹理，所以使用同一根毛皮

		        //Add the first triangle of the fin quad
		        // 添加鳍四边形的第一个三角形
		        gl_Position = ubo.proj * ubo.view * ubo.model * gl_in[i].gl_Position;
		        EmitVertex();
                
		        fragAlpha = 1;
		        fragColor = vec3(1.0,0.0,0.0);	//Red silhouettte //红色轮廓
		        fragTexCoord = vec2(1.0f,0.0f);	//Because of lack of good fin texture use the same fur one	 由于缺乏良好的鳍状纹理，所以使用同一根毛皮

		        gl_Position = ubo.proj * ubo.view * ubo.model * (gl_in[prevVer].gl_Position+vec4(inNormal[prevVer]*FUR_LENGTH,0.0));
		        EmitVertex();

		        fragAlpha = 1;
		        fragColor = vec3(1.0,0.0,0.0);	//Red silhouettte //红色轮廓
		        fragTexCoord = vec2(1.0f,1.0f);	//Because of lack of good fin texture use the same fur one	 由于缺乏良好的鳍状纹理，所以使用同一根毛皮
		        gl_Position = ubo.proj * ubo.view * ubo.model * gl_in[prevVer].gl_Position;
		        EmitVertex();

		        EndPrimitive();


	            //Add the second triangle of the fin quad
		        //添加鳍四边形的第二个三角形

		        fragAlpha = 1;
		        fragColor = vec3(1.0,0.0,0.0);	//Red silhouettte //红色轮廓
		        fragTexCoord = vec2(0.0f,0.0f);	//Because of lack of good fin texture use the same fur one	 由于缺乏良好的鳍状纹理，所以使用同一根毛皮
				
			
		        gl_Position = ubo.proj * ubo.view * ubo.model * (gl_in[i].gl_Position+vec4(inNormal[i]*FUR_LENGTH,0.0));
		        EmitVertex();


                fragAlpha = 1;
		        fragColor = vec3(1.0,0.0,0.0);	//Red silhouettte //红色轮廓
		        fragTexCoord = vec2(1.0f,0.0f);	//Because of lack of good fin texture use the same fur one	 由于缺乏良好的鳍状纹理，所以使用同一根毛皮

		        gl_Position = ubo.proj * ubo.view * ubo.model * (gl_in[prevVer].gl_Position+vec4(inNormal[prevVer]*FUR_LENGTH,0.0));
		        EmitVertex();


                fragAlpha = 1;
		        fragColor = vec3(1.0,0.0,0.0);	//Red silhouettte //红色轮廓
		        fragTexCoord = vec2(0.0f,1.0f);	//Because of lack of good fin texture use the same fur one	 由于缺乏良好的鳍状纹理，所以使用同一根毛皮


		        gl_Position = ubo.proj * ubo.view * ubo.model * gl_in[i].gl_Position;
		        EmitVertex();
	
		        EndPrimitive();
			}
		}


		
	}
	


}
