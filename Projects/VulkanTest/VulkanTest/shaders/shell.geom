#version 450

layout(binding = 0) uniform UniformBufferObject 
{
    mat4 model;
    mat4 view;
    mat4 proj;
	
} ubo;

layout(binding = 1) uniform OffsetUniformBufferObject 
{
    float alpha;
    float len;
	
} Offset;


in gl_PerVertex
{
    vec4  gl_Position;
  
} gl_in[];

//输入

layout(triangles_adjacency) in;

layout(location = 0) in vec3 inColor[6];
layout(location = 1) in vec3 inNormal[6];


//输出

layout(triangle_strip, max_vertices = 30) out;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out float fragAlpha;

const vec3 Center = vec3(-0.016751,0.109869,-0.00129);

void main() 
{
	//Go through the actual vetices in positions 0,2,4 because of the topology
	for(int i = 0; i < gl_in.length(); i+=2)
	{
		fragAlpha = Offset.alpha;		

		fragColor = inColor[i];			
		fragTexCoord = inTexCoord[i];

		vec3 PosInMesh = gl_in[i].gl_Position.xyz + inNormal[i].xyz * Offset.len;
		vec3 VertexVector = PosInMesh - Center;

	


        vec3 n = normalize(VertexVector);
        float u = atan(n.x, n.z) / (2*3.1415926) + 0.5;
        float v = n.y * 0.5 + 0.5;
	


		fragTexCoord = vec2(u,v);

		//Add a vertex away from the original one
		gl_Position = ubo.proj * ubo.view * ubo.model * (gl_in[i].gl_Position + vec4(inNormal[i] * Offset.len, 0.0));

		EmitVertex();
	}
	//Create a primitive from the vertices
	EndPrimitive();


}
