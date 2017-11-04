//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Tóth András
// Neptun : AZLU13
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================


#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

#define IMAGE 0
#define CPU_PROCEDURAL 1
#define GPU_PROCEDURAL 2

#define TEXTURING CPU_PROCEDURAL

const unsigned int windowWidth = 600, windowHeight = 600;


#if TEXTURING==IMAGE
byte* ReadBMP(char * pathname, int& width, int& height) {	// read image as BMP files 
	FILE * file = fopen(pathname, "r");
	if (!file) {
		printf("%s does not exist\n", pathname);
		return NULL;
	}
	unsigned short bitmapFileHeader[27];					// bitmap header
	fread(&bitmapFileHeader, 27, 2, file);
	if (bitmapFileHeader[0] != 0x4D42) {   // magic number
		return NULL;
	}
	if (bitmapFileHeader[14] != 24) {
		printf("only true color bmp files are supported\n");
		return NULL;
	}
	width = bitmapFileHeader[9];
	height = bitmapFileHeader[11];
	unsigned int size = (unsigned long)bitmapFileHeader[17] + (unsigned long)bitmapFileHeader[18] * 65536;
	fseek(file, 54, SEEK_SET);

	byte* image = new byte[size];
	fread(image, 1, size, file); 	// read the pixels

									// Swap R and B since in BMP, the order is BGR
	for (int imageIdx = 0; imageIdx < size; imageIdx += 3) {
		byte tempRGB = image[imageIdx];
		image[imageIdx] = image[imageIdx + 2];
		image[imageIdx + 2] = tempRGB;
	}
	return image;
}
#endif

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec2 vertexUV;			// Attrib Array 1

	out vec2 texCoord;								// output attribute

	void main() {
		texCoord = vertexUV;														// copy texture coordinates
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
#if TEXTURING==IMAGE || TEXTURING==CPU_PROCEDURAL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	uniform vec2 texCursor;

	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

    vec2 LensDistortion(vec2 p) {
		const float maxRadius2 = 0.1f; 
		float radius2 = dot(texCoord - texCursor, texCoord - texCursor);
		float scale = (radius2 < maxRadius2) ? radius2 / maxRadius2 : 1;
		return (texCoord - texCursor) * scale + texCursor;
	}

	void main() {
		fragmentColor = texture(textureUnit, texCoord);
	}
)";
#endif

#if TEXTURING==GPU_PROCEDURAL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	uniform vec2 texCursor;

	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

    vec2 LensDistortion(vec2 p) {
		const float maxRadius2 = 0.1f; 
		float radius2 = dot(texCoord - texCursor, texCoord - texCursor);
		float scale = (radius2 < maxRadius2) ? radius2 / maxRadius2 : 1;
		return (texCoord - texCursor) * scale + texCursor;
	}

    int Mandelbrot(vec2 c) {
		vec2 p = c;
		for(int i = 10000; i > 0; i--) {
			p = vec2(p.x * p.x - p.y * p.y + c.x, 2 * p.x * p.y + c.y); // z_{n+1} = z_{n}^2 + c
			if (dot(p, p) > 4) return i;
		}
		return 0;
	}

	void main() {
		int i = Mandelbrot(LensDistortion(texCoord) * 3 - vec2(2, 1.5)); 
		fragmentColor = vec4((i % 5)/5.0f, (i % 11) / 11.0f, (i % 31) / 31.0f, 1); 
	}
)";
#endif

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}
};

// 3D point in Cartesian coordinates or RGB color
struct vec3 {
	float v[3];

	vec3(float x = 0, float y = 0, float z = 0) { v[0] = x; v[1] = y; v[2] = z; }
	vec3(const vec4& hom) { v[0] = hom.v[0] / hom.v[3]; v[1] = hom.v[1] / hom.v[3]; v[2] = hom.v[2] / hom.v[3]; }
	vec3 operator*(float s) {
		vec3 res(v[0] * s, v[1] * s, v[2] * s);
		return res;
	}

	vec3& operator+=(vec3 a) {
		for (int i = 0; i < 3; i++) {
			v[i] += a.v[i];
		}
		return *this;
	}

	float operator*(vec3 a) {
		float s = 0;
		for (int i = 0; i < 3; i++) {
			s += v[i] * a.v[i];
		}
		return s;
	}

	vec3 operator-(vec3 a) {
		vec3 s;
		for (int i = 0; i < 3; i++) {
			s.v[i] = v[i] - a.v[i];
		}
		return s;
	}

	float length() {
		float l = 0;
		for (int i = 0; i < 3; i++) {
			l += v[i] * v[i];
		}
		return sqrtf(l);
	}
};

// 2D point in Cartesian coordinates
struct vec2 {
	float v[2];

	vec2(float x = 0, float y = 0) { v[0] = x; v[1] = y; }
	vec2(const vec4& hom) { v[0] = hom.v[0] / hom.v[3]; v[1] = hom.v[1] / hom.v[3]; }
	vec2 operator-(vec2& right) {
		vec2 res(v[0] - right.v[0], v[1] - right.v[1]);
		return res;
	}
	float Length() { return sqrtf(v[0] * v[0] + v[1] * v[1]); }
};


// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = 0; // 10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

vec2 texLensPosition(2, 2);

class Torus {
	float r, R;
public:
	Torus() {
		r = 1;
		R = 4;
	}

	vec3 transform(vec2 uv) {
		vec3 xyz;
		xyz.v[0]=  (R+r*cosf(uv.v[0]))*cosf(uv.v[1]);
		xyz.v[1] = (R + r*cosf(uv.v[0]))*sinf(uv.v[1]);
		xyz.v[2] = r*sinf(uv.v[0]);
		return xyz;
	}

	float gauss(vec2 uv) {
		return (1 / r)*(cosf(uv.v[0]) / (R + r*cosf(uv.v[0])));
	}
};
Torus torus;

class TexturedQuad {
	unsigned int vao, vbo[2], textureId;	// vertex array object id and texture id
	vec2 vertices[4], uvs[4];
public:
	TexturedQuad() {
		vertices[0] = vec2(-10, -10); uvs[0] = 
			(0, 0);
		vertices[1] = vec2(10, -10);  uvs[1] = vec2(1, 0);
		vertices[2] = vec2(10, 10);   uvs[2] = vec2(1, 1);
		vertices[3] = vec2(-10, 10);  uvs[3] = vec2(0, 1);
	}
	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		glGenBuffers(2, vbo);	// Generate 1 vertex buffer objects

								// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);	   // copy to that part of the memory which is not modified 
																					   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
																			   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		float maxred = torus.gauss(vec2(0, M_PI));
		float maxgreen = torus.gauss(vec2(M_PI, M_PI));
		int width = windowWidth, height = windowHeight;
		vec3 * image = new vec3[width * height];
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float g = torus.gauss(vec2(float(x) / width * 2 * M_PI, float(y) / height * 2 * M_PI));
				float red = (g > 0) ? g : 0;
				float green = (g < 0) ? g : 0;
				
				image[y * width + x] = vec3(red/maxred, green/maxgreen, 0);
			}
		}


		// Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGB, GL_FLOAT, image); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // sampling

	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source

		mat4 MVPTransform = camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		location = glGetUniformLocation(shaderProgram, "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
		}


		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

class LineStrip {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[500]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
	unsigned int textureId;
public:
	LineStrip() {
		nVertices = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

		vec3 image[1];
		image[0] = vec3(1, 1, 1);

		// Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGB, GL_FLOAT, image); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // sampling
	}

	void AddPoint(float cX, float cY) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		if (nVertices >= 100) return;

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[5 * nVertices] = wVertex.v[0];
		vertexData[5 * nVertices + 1] = wVertex.v[1];
		nVertices++;
		// copy data to the GPU
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) {
				glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
				glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
			}
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nVertices);
		}
	}
};

class BezierCurve {
	int n;
	LineStrip line;
	std::vector<vec3> cps;	// control points
	float B(int i, float t) {
		int n = cps.size() - 1; // n deg polynomial = n+1 pts! 
		float choose = 1;
		for (int j = 1; j <= i; j++)choose *= (float)(n - j + 1) / j;
		return choose * pow(t, i) * pow(1 - t, n - i);
	} public:
		BezierCurve() {
			n = 0;
		}

		void AddControlPoint(vec3 cp) {
			cps.push_back(cp);
			n++;

			line = LineStrip();
			line.Create();

			for (int t = 0; t < 19; t++) {
				vec3 temp = r(t / float(19));
				line.AddPoint(temp.v[0], temp.v[1]);
			}
			vec3 end = cps[n - 1];
			line.AddPoint(end.v[0], end.v[1]);
		}

		vec3 r(float t) {
			vec3 rr(0, 0);
			for (int i = 0; i < cps.size(); i++) rr += cps[i] * B(i, t);
			return rr;
		}

		void Draw() {
			line.Draw();
		}
};

float angle(vec3 a, vec3 b) {
	return  abs(acosf(a * b / (a.length()*b.length())) * 180 / M_PI);
}
class Triangle {
	BezierCurve bezier[3];
	std::vector<vec3> cps;
public:
	void AddControlPoint(vec3 cp) {
		if (cps.size()==3){
			for (int i = 0; i < 3; i++) {
				bezier[i] = BezierCurve();
			}
			cps.clear();
		}
		
		cps.push_back(cp);
		
		if (cps.size() == 3) {
			vec3 cp01 = AddBrezierControlpoint(0, 1);
			vec3 cp12 = AddBrezierControlpoint(1, 2);
			vec3 cp20 = AddBrezierControlpoint(2, 0);

			float sum=0;
			sum += angle(cp01 - cps[0], cp20 - cps[0]);
			sum += angle(cp01 - cps[1], cp12 - cps[1]);
			sum += angle(cp20 - cps[2], cp12 - cps[2]);
			printf("%f\n", sum);

		}
	}

	vec3 AddBrezierControlpoint(int first, int second) {
		bezier[first].AddControlPoint(cps[first]);
		vec3 cp = BestPoint(cps[first], cps[second]);
		bezier[first].AddControlPoint(cp);
		bezier[first].AddControlPoint(cps[second]);
		return cp;
	}

	vec3 BestPoint(vec3 A, vec3 B) {
		vec3 p;
		p.v[0] = (A.v[0] + B.v[0]) / 2;
		p.v[1] = (A.v[1] + B.v[1]) / 2;

		float E = Length(A, p, B);
		for (int i = 0; i < 100; i++) {
			vec3 np = Perturb(p);
			float nE = Length(A, np, B);
			if (nE < E) {
				E = nE;
				p = np;							//Kérem az 1 millió dollárt
			}
		}

		return p;
	}

	float Length(vec3 A, vec3 B, vec3 C) {
		BezierCurve bc;
		bc.AddControlPoint(A);
		bc.AddControlPoint(B);
		bc.AddControlPoint(C);

		const int n=10;
		vec3 temp[n];
		for (int i = 0; i < n-1; i++) {
			vec3 vec = bc.r(float(i) / 10);
			vec.v[0] = (vec.v[0] + 1)*M_PI;
			vec.v[1] = (vec.v[1] + 1)*M_PI;
			temp[i] = torus.transform(vec2(vec.v[0], vec.v[1]));
		}
		C.v[0] = (C.v[0] + 1)*M_PI;
		C.v[1] = (C.v[1] + 1)*M_PI;
		temp[n-1] = torus.transform(vec2(C.v[0], C.v[1]));

		float d = 0;
		for (int i = 0; i < n - 1; i++) {
			float pd = 0;
			for (int j = 0; j < 3; j++) {
				pd += (temp[i].v[j] - temp[i + 1].v[j])*(temp[i].v[j] - temp[i + 1].v[j]);
			}
			d += sqrtf(pd);
		}
			
		return d;
	}

	vec3 Perturb(vec3 P) {
		vec3 t;
		float x = P.v[0] - 1 + (rand() % 100)*0.02;
		float y = P.v[1] - 1 + (rand() % 100)*0.02;
		return vec3(x, y, 0);
	}

	void Draw() {
		for (int i = 0; i < 3; i++) {
			bezier[i].Draw();
		}
	}
};


// The virtual world: collection of three objects
TexturedQuad quad;
Triangle triangle;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	quad.Create();

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	quad.Draw();
	triangle.Draw();

	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}


// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		triangle.AddControlPoint(vec3(cX, cY, 0));
		glutPostRedisplay();     // redraw
	}
}


// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
