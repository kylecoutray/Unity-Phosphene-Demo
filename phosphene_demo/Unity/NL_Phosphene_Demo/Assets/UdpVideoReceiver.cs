using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

public class UdpVideoReceiver : MonoBehaviour
{
    public string udpHost = "127.0.0.1";
    public int udpPort = 5556;
    public bool logUdpStatus = true;
    public Renderer targetRenderer;
    public string textureProperty = "_MainTex";
    public RawImage targetImage;

    UdpClient udpClient;
    Thread udpThread;
    volatile bool running;

    readonly Queue<byte[]> packetQueue = new Queue<byte[]>();
    readonly object packetLock = new object();
    readonly Dictionary<uint, FrameAssembly> assemblies = new Dictionary<uint, FrameAssembly>();

    Texture2D texture;
    bool textureDirty;

    const int HeaderSize = 26;
    const uint MagicVideo = 0x56494430; // "VID0"
    const byte FormatJpeg = 10;

    class FrameAssembly
    {
        public byte[] Buffer;
        public int PayloadLength;
        public ushort TotalChunks;
        public ushort ChunksReceived;
        public bool[] ChunkFlags;
        public int Width;
        public int Height;
        public byte Format;
        public float LastTouched;
        public uint FrameId;
    }

    void Awake()
    {
        if (targetRenderer == null)
        {
            targetRenderer = GetComponent<Renderer>();
        }
        if (targetImage == null)
        {
            targetImage = GetComponent<RawImage>();
        }
    }

    void OnEnable()
    {
        StartUdpListener();
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
        lock (packetLock)
        {
            packetQueue.Clear();
        }
        assemblies.Clear();
    }

    void Update()
    {
        bool newFrame = TryProcessPackets();
        if (newFrame && textureDirty)
        {
            textureDirty = false;
            if (targetRenderer != null)
            {
                targetRenderer.material.SetTexture(textureProperty, texture);
            }
            if (targetImage != null)
            {
                targetImage.texture = texture;
            }
        }
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
                Debug.LogFormat("[UdpVideoReceiver] Listening for UDP video on {0}:{1}", udpHost, udpPort);
            }
        }
        catch (Exception ex)
        {
            Debug.LogError("[UdpVideoReceiver] Failed to start UDP listener: " + ex.Message);
            running = false;
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

    bool TryProcessPackets()
    {
        bool applied = false;
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
            applied |= ProcessPacket(packet);
        }

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

        return applied;
    }

    bool ProcessPacket(byte[] data)
    {
        if (data == null || data.Length < HeaderSize)
        {
            return false;
        }

        int offset = 0;
        uint magic = BinaryPrimitives.ReadUInt32BigEndian(data.AsSpan(offset, 4));
        offset += 4;
        if (magic != MagicVideo)
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
        ushort width = BinaryPrimitives.ReadUInt16BigEndian(data.AsSpan(offset, 2));
        offset += 2;
        ushort height = BinaryPrimitives.ReadUInt16BigEndian(data.AsSpan(offset, 2));
        offset += 2;
        byte fmtCode = data[offset++];
        byte _ = data[offset++];

        int chunkLen = data.Length - HeaderSize;
        if (chunkLen <= 0 || chunkOffset + (uint)chunkLen > payloadLength)
        {
            return false;
        }

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
                Width = width,
                Height = height,
                Format = fmtCode,
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
            Buffer.BlockCopy(data, HeaderSize, assembly.Buffer, (int)chunkOffset, chunkLen);
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
        if (assembly.Format != FormatJpeg)
        {
            return false;
        }

        if (texture == null)
        {
            texture = new Texture2D(2, 2, TextureFormat.RGB24, false);
            texture.wrapMode = TextureWrapMode.Clamp;
            texture.filterMode = FilterMode.Bilinear;
        }

        if (!texture.LoadImage(assembly.Buffer, false))
        {
            return false;
        }

        textureDirty = true;
        return true;
    }
}
