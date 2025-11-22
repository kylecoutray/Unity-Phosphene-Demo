Shader "Custom/PhospheneGaussian"
{
    Properties
    {
        _Sigma ("Sigma", Float) = 0.06
        _Source ("Source (debug)", 2D) = "white" {}
        _Intensity ("Global Intensity", Float) = 1.0
        _Gamma ("Output Gamma", Float) = 1.0
        _GridSize ("Grid Size", Float) = 32.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Transparent" }
        Cull Off
        ZWrite Off
        Blend One One

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5

            #include "UnityCG.cginc"

            StructuredBuffer<float2> _Points;
            StructuredBuffer<float> _Weights;
            sampler2D _Source;

            float _Sigma;
            float _Intensity;
            float _Gamma;
            int _GridSize;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float gaussian(float2 uv, float2 center, float sigma)
            {
                float2 mirroredUV = float2(1.0 - uv.x, uv.y);
                float2 d = mirroredUV - center;
                float dist2 = dot(d, d);
                float s2 = max(1e-6, sigma * sigma);
                return exp(-0.5 * dist2 / s2);
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float sigma = _Sigma;
                float energy = 0.0;
                int total = _GridSize * _GridSize;
                [loop]
                for (int idx = 0; idx < total; ++idx)
                {
                    float2 center = _Points[idx];
                    float weight = _Weights[idx];
                    energy += weight * gaussian(i.uv, center, sigma);
                }
                float norm = max(1e-5, (float)total);
                energy = energy / norm;
                energy = saturate(energy * _Intensity);
                if (_Gamma != 1.0)
                {
                    energy = pow(energy, _Gamma);
                }
                return fixed4(energy, energy, energy, energy);
            }
            ENDCG
        }
    }
}
