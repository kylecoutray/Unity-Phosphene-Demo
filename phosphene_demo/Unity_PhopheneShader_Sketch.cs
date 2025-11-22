// Unity phosphene renderer sketch (attach to a Quad with a RenderTexture target)
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public class PhospheneRenderer : MonoBehaviour
{
    public int grid = 32;
    public float sigma = 0.06f;
    public Texture2D cameraTex; // optional fallback feed from WebCamTexture or AR camera
    public Material mat;        // shader should sum Gaussians from a buffer
    [Range(0.1f, 200f)]
    public float shaderIntensity = 40f;
    [Range(0.2f, 3f)]
    public float shaderGamma = 1f;

    [Header("UDP input from Python phosphene_demo.py")]
    public bool useUdp = true;
    public string udpHost = "127.0.0.1";
    public int udpPort = 5555;
    public PayloadFormat udpFormat = PayloadFormat.Float32;
    public bool logUdpStatus = true;

    public enum PayloadFormat { Float32, UInt8 }

    ComputeBuffer pointsBuf;
    ComputeBuffer weightsBuf;

    UdpClient udpClient;
    Thread udpThread;
    volatile bool running;
    readonly Queue<byte[]> packetQueue = new Queue<byte[]>();
    readonly object packetLock = new object();

    float[] tmpWeights;
    readonly Dictionary<uint, FrameAssembly> assemblies = new Dictionary<uint, FrameAssembly>();
    const int UdpHeaderSize = 26;
    const uint UdpMagic = 0x50484F53; // "PHOS"

    class FrameAssembly
    {
        public byte[] Buffer;
        public int PayloadLength;
        public ushort TotalChunks;
        public ushort ChunksReceived;
        public bool[] ChunkFlags;
        public int GridSize;
        public PayloadFormat Format;
        public float LastTouched;
        public uint FrameId;
    }

    void OnEnable()
    {
        ConfigureGrid(grid);
        if (useUdp)
        {
            StartUdpListener();
        }
    }

    void Start()
    {
        tmpWeights = new float[grid * grid];
    }

    void ConfigureGrid(int newGrid)
    {
        grid = Mathf.Max(2, newGrid);
        Vector2[] pts = new Vector2[grid * grid];
        int k = 0;
        for (int y = 0; y < grid; y++)
        {
            for (int x = 0; x < grid; x++)
            {
                pts[k++] = new Vector2((x + 0.5f) / grid, (y + 0.5f) / grid);
            }
        }
        pointsBuf?.Dispose();
        weightsBuf?.Dispose();
        pointsBuf = new ComputeBuffer(pts.Length, sizeof(float) * 2);
        pointsBuf.SetData(pts);
        weightsBuf = new ComputeBuffer(pts.Length, sizeof(float));
        if (mat != null)
        {
            mat.SetBuffer("_Points", pointsBuf);
            mat.SetBuffer("_Weights", weightsBuf);
            mat.SetFloat("_Sigma", sigma);
            mat.SetInt("_GridSize", grid);
            mat.SetFloat("_Intensity", shaderIntensity);
            mat.SetFloat("_Gamma", shaderGamma);
        }
        tmpWeights = new float[pts.Length];
    }

    void StartUdpListener()
    {
        try
        {
            IPAddress ip;
            if (string.IsNullOrWhiteSpace(udpHost) || udpHost == "*" || udpHost.ToLowerInvariant() == "any")
            {
                ip = IPAddress.Any;
            }
            else
            {
                ip = IPAddress.Parse(udpHost);
            }
            udpClient = new UdpClient(new IPEndPoint(ip, udpPort));
            running = true;
            udpThread = new Thread(UdpListenLoop) { IsBackground = true };
            udpThread.Start();
            if (logUdpStatus)
            {
                Debug.LogFormat("[PhospheneRenderer] Listening for UDP packets on {0}:{1} ({2})",
                    udpHost, udpPort, udpFormat);
            }
        }
        catch (Exception ex)
        {
            Debug.LogError("[PhospheneRenderer] Failed to start UDP listener: " + ex.Message);
            useUdp = false;
        }
    }

    void UdpListenLoop()
    {
        IPEndPoint endpoint = new IPEndPoint(IPAddress.Any, 0);
        while (running)
        {
            try
            {
                byte[] data = udpClient.Receive(ref endpoint);
                lock (packetLock)
                {
                    if (packetQueue.Count > 512)
                    {
                        packetQueue.Dequeue();
                    }
                    packetQueue.Enqueue(data);
                }
            }
            catch (SocketException)
            {
                Thread.Sleep(5);
            }
            catch (ObjectDisposedException)
            {
                break;
            }
        }
    }

    bool TryDecodePacket()
    {
        bool newFrameApplied = false;
        while (true)
        {
            byte[] packet = null;
            lock (packetLock)
            {
                if (packetQueue.Count > 0)
                {
                    packet = packetQueue.Dequeue();
                }
            }
            if (packet == null)
            {
                break;
            }
            newFrameApplied |= ProcessPacket(packet);
        }

        // Clean stale assemblies
        if (assemblies.Count > 0)
        {
            float now = Time.time;
            List<uint> toRemove = null;
            foreach (var kv in assemblies)
            {
                if (now - kv.Value.LastTouched > 0.5f)
                {
                    toRemove ??= new List<uint>();
                    toRemove.Add(kv.Key);
                }
            }
            if (toRemove != null)
            {
                foreach (var id in toRemove)
                {
                    assemblies.Remove(id);
                }
            }
        }

        return newFrameApplied;
    }

    bool ProcessPacket(byte[] data)
    {
        if (data == null || data.Length < UdpHeaderSize)
        {
            return false;
        }

        int offset = 0;
        uint magic = BinaryPrimitives.ReadUInt32BigEndian(data.AsSpan(offset, 4));
        offset += 4;
        if (magic != UdpMagic)
        {
            return false;
        }

        uint frameId = BinaryPrimitives.ReadUInt32BigEndian(data.AsSpan(offset, 4));
        offset += 4;
        ushort totalChunks = BinaryPrimitives.ReadUInt16BigEndian(data.AsSpan(offset, 2));
        offset += 2;
        ushort chunkIndex = BinaryPrimitives.ReadUInt16BigEndian(data.AsSpan(offset, 2));
        offset += 2;
        uint payloadLength = BinaryPrimitives.ReadUInt32BigEndian(data.AsSpan(offset, 4));
        offset += 4;
        uint chunkOffset = BinaryPrimitives.ReadUInt32BigEndian(data.AsSpan(offset, 4));
        offset += 4;
        ushort dim0 = BinaryPrimitives.ReadUInt16BigEndian(data.AsSpan(offset, 2));
        offset += 2;
        ushort dim1 = BinaryPrimitives.ReadUInt16BigEndian(data.AsSpan(offset, 2));
        offset += 2;
        byte fmtCode = data[offset++];
        byte _ = data[offset++]; // flags/reserved

        int chunkLen = data.Length - UdpHeaderSize;
        if (chunkLen <= 0 || chunkOffset + (uint)chunkLen > payloadLength)
        {
            return false;
        }

        PayloadFormat fmt = fmtCode == 1 ? PayloadFormat.Float32 : PayloadFormat.UInt8;

        FrameAssembly assembly;
        if (!assemblies.TryGetValue(frameId, out assembly))
        {
            assembly = new FrameAssembly
            {
                Buffer = new byte[payloadLength],
                PayloadLength = (int)payloadLength,
                TotalChunks = totalChunks,
                ChunkFlags = new bool[totalChunks],
                ChunksReceived = 0,
                GridSize = dim0,
                Format = fmt,
                LastTouched = Time.time,
                FrameId = frameId
            };
            assemblies[frameId] = assembly;
        }

        if (assembly.PayloadLength != payloadLength || assembly.TotalChunks != totalChunks)
        {
            assemblies.Remove(frameId);
            return false;
        }

        if (!assembly.ChunkFlags[chunkIndex])
        {
            Buffer.BlockCopy(data, UdpHeaderSize, assembly.Buffer, (int)chunkOffset, chunkLen);
            assembly.ChunkFlags[chunkIndex] = true;
            assembly.ChunksReceived++;
            assembly.LastTouched = Time.time;
        }

        if (assembly.ChunksReceived >= assembly.TotalChunks)
        {
            bool applied = ApplyFrame(assembly);
            assemblies.Remove(frameId);
            return applied;
        }

        return false;
    }

    bool ApplyFrame(FrameAssembly assembly)
    {
        int expectedElements = assembly.GridSize * assembly.GridSize;
        int elementSize = assembly.Format == PayloadFormat.Float32 ? sizeof(float) : sizeof(byte);
        if (assembly.PayloadLength != expectedElements * elementSize)
        {
            return false;
        }

        if (assembly.GridSize != grid)
        {
            ConfigureGrid(assembly.GridSize);
            if (logUdpStatus)
            {
                Debug.LogFormat("[PhospheneRenderer] Adjusted grid to {0}x{0}.", assembly.GridSize);
            }
        }

        int count = assembly.GridSize * assembly.GridSize;
        if (tmpWeights == null || tmpWeights.Length != count)
        {
            tmpWeights = new float[count];
        }

        if (assembly.Format == PayloadFormat.Float32)
        {
            Buffer.BlockCopy(assembly.Buffer, 0, tmpWeights, 0, assembly.PayloadLength);
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                tmpWeights[i] = assembly.Buffer[i] / 255.0f;
            }
        }
        udpFormat = assembly.Format;

        return true;
    }

    void Update()
    {
        bool hasWeights = false;
        if (useUdp)
        {
            hasWeights = TryDecodePacket();
        }

        if (!hasWeights && cameraTex != null)
        {
            // Fallback: downsample the camera texture locally.
            for (int gy = 0; gy < grid; gy++)
            {
                for (int gx = 0; gx < grid; gx++)
                {
                    float acc = 0f; int cnt = 0;
                    int x0 = gx * cameraTex.width / grid;
                    int x1 = (gx + 1) * cameraTex.width / grid;
                    int y0 = gy * cameraTex.height / grid;
                    int y1 = (gy + 1) * cameraTex.height / grid;
                    for (int y = y0; y < y1; y++)
                    {
                        for (int x = x0; x < x1; x++)
                        {
                            acc += cameraTex.GetPixel(x, y).grayscale;
                            cnt++;
                        }
                    }
                    tmpWeights[gy * grid + gx] = Mathf.Pow(acc / Mathf.Max(1, cnt), 0.8f);
                }
            }
            hasWeights = true;
        }
        if (hasWeights)
        {
            weightsBuf.SetData(tmpWeights);
        }
        if (mat != null)
        {
            mat.SetFloat("_Sigma", sigma);
            mat.SetInt("_GridSize", grid);
            mat.SetFloat("_Intensity", shaderIntensity);
            mat.SetFloat("_Gamma", shaderGamma);
            if (cameraTex != null)
            {
                mat.SetTexture("_Source", cameraTex);
            }
        }
    }

    void OnDisable()
    {
        running = false;
        udpClient?.Close();
        udpClient = null;
        if (udpThread != null && udpThread.IsAlive)
        {
            udpThread.Join(100);
            udpThread = null;
        }
        pointsBuf?.Dispose();
        weightsBuf?.Dispose();
        pointsBuf = null;
        weightsBuf = null;
        lock (packetLock)
        {
            packetQueue.Clear();
        }
        assemblies.Clear();
    }

    void OnDestroy()
    {
        pointsBuf = null;
        weightsBuf = null;
    }
}
