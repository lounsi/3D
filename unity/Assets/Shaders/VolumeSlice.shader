// BrainXR — VolumeSlice.shader
// Shader custom avec support transparence et plan de coupe

Shader "BrainXR/VolumeSlice"
{
    Properties
    {
        _Color ("Color", Color) = (0.85, 0.75, 0.75, 1)
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _Glossiness ("Smoothness", Range(0,1)) = 0.5
        _Metallic ("Metallic", Range(0,1)) = 0.0
        _ClipPlane ("Clip Plane", Vector) = (0, 1, 0, 0)
        [Toggle(_CLIP_ENABLED)] _ClipEnabled ("Enable Clipping", Float) = 0
        _CrossSectionColor ("Cross Section Color", Color) = (1, 0.5, 0.5, 1)
    }
    
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        LOD 200
        Cull Off  // Afficher les deux faces pour la coupe
        
        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows alpha:fade
        #pragma shader_feature _CLIP_ENABLED
        #pragma target 3.0
        
        sampler2D _MainTex;
        half _Glossiness;
        half _Metallic;
        fixed4 _Color;
        float4 _ClipPlane;
        fixed4 _CrossSectionColor;
        
        struct Input
        {
            float2 uv_MainTex;
            float3 worldPos;
            float facing : VFACE;
        };
        
        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            #if _CLIP_ENABLED
            // Plan de coupe : clip les pixels au-dessus du plan
            float dist = dot(IN.worldPos, _ClipPlane.xyz) + _ClipPlane.w;
            clip(-dist);
            #endif
            
            fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * _Color;
            
            // Face arriere (coupe) : couleur differente
            if (IN.facing < 0)
            {
                o.Albedo = _CrossSectionColor.rgb;
                o.Emission = _CrossSectionColor.rgb * 0.3;
            }
            else
            {
                o.Albedo = c.rgb;
            }
            
            o.Metallic = _Metallic;
            o.Smoothness = _Glossiness;
            o.Alpha = _Color.a;
        }
        ENDCG
    }
    
    FallBack "Standard"
}
