import React, { useRef, useEffect, useState } from 'react'

const MusicVisualizer = ({ 
  audioRef, 
  duration, 
  currentTime, 
  beats = [], 
  melodySegments = [], 
  tangoSections = [],
  visualizationMode = 'beats', // 'beats' or 'melody'
  onSeek 
}) => {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const [canvasWidth, setCanvasWidth] = useState(800)
  const [isHovering, setIsHovering] = useState(false)
  const [hoverTime, setHoverTime] = useState(0)

  // Update canvas width on resize
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setCanvasWidth(containerRef.current.offsetWidth)
      }
    }
    
    updateWidth()
    window.addEventListener('resize', updateWidth)
    return () => window.removeEventListener('resize', updateWidth)
  }, [])

  // Draw visualization
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || duration === 0) return

    const ctx = canvas.getContext('2d')
    const height = canvas.height
    const width = canvas.width

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw background
    ctx.fillStyle = '#f8fafc'
    ctx.fillRect(0, 0, width, height)

    // Draw progress background
    ctx.fillStyle = '#e2e8f0'
    ctx.fillRect(0, height - 8, width, 8)

    // Draw progress
    const progressWidth = (currentTime / duration) * width
    ctx.fillStyle = '#3b82f6'
    ctx.fillRect(0, height - 8, progressWidth, 8)

    if (visualizationMode === 'beats') {
      // Draw beats
      beats.forEach(beat => {
        const x = (beat.time / duration) * width
        const intensity = beat.confidence || 0.5
        
        // Beat line
        ctx.strokeStyle = `rgba(239, 68, 68, ${0.3 + intensity * 0.7})`
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(x, 10)
        ctx.lineTo(x, height - 20)
        ctx.stroke()

        // Beat marker
        ctx.fillStyle = `rgba(239, 68, 68, ${0.5 + intensity * 0.5})`
        ctx.beginPath()
        ctx.arc(x, height - 12, 3 + intensity * 2, 0, 2 * Math.PI)
        ctx.fill()
      })
    } else if (visualizationMode === 'melody') {
      // Draw melody segments
      const centerY = height / 2
      
      melodySegments.forEach((segment, index) => {
        const startX = (segment.start / duration) * width
        const endX = (segment.end / duration) * width
        const segmentWidth = endX - startX
        
        if (segmentWidth < 1) return // Skip very small segments

        const intensity = segment.confidence || 0.5
        
        if (segment.type === 'staccato') {
          // Spiky line for staccato
          ctx.strokeStyle = `rgba(168, 85, 247, ${0.4 + intensity * 0.6})`
          ctx.lineWidth = 2
          ctx.beginPath()
          
          const spikes = Math.max(3, Math.floor(segmentWidth / 10))
          for (let i = 0; i <= spikes; i++) {
            const x = startX + (i / spikes) * segmentWidth
            const y = i % 2 === 0 ? centerY - 15 - intensity * 10 : centerY + 15 + intensity * 10
            if (i === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
          }
          ctx.stroke()
        } else {
          // Smooth line for legato
          ctx.strokeStyle = `rgba(34, 197, 94, ${0.4 + intensity * 0.6})`
          ctx.lineWidth = 3
          ctx.beginPath()
          
          const amplitude = 10 + intensity * 15
          const frequency = 0.02
          
          for (let x = startX; x <= endX; x += 2) {
            const progress = (x - startX) / segmentWidth
            const y = centerY + Math.sin(progress * Math.PI * 2 * frequency * segmentWidth) * amplitude
            if (x === startX) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
          }
          ctx.stroke()
        }

        // Segment background
        ctx.fillStyle = segment.type === 'staccato' 
          ? `rgba(168, 85, 247, 0.1)` 
          : `rgba(34, 197, 94, 0.1)`
        ctx.fillRect(startX, 10, segmentWidth, height - 30)
      })
    }

    // Draw current time indicator
    const currentX = (currentTime / duration) * width
    ctx.strokeStyle = '#1f2937'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(currentX, 0)
    ctx.lineTo(currentX, height - 8)
    ctx.stroke()

    // Draw hover indicator
    if (isHovering) {
      const hoverX = (hoverTime / duration) * width
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)'
      ctx.lineWidth = 1
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.moveTo(hoverX, 0)
      ctx.lineTo(hoverX, height - 8)
      ctx.stroke()
      ctx.setLineDash([])
    }

  }, [duration, currentTime, beats, melodySegments, visualizationMode, canvasWidth, isHovering, hoverTime])

  const handleMouseMove = (e) => {
    if (!containerRef.current || duration === 0) return
    
    const rect = containerRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const time = (x / canvasWidth) * duration
    setHoverTime(Math.max(0, Math.min(duration, time)))
  }

  const handleClick = (e) => {
    if (!containerRef.current || duration === 0) return
    
    const rect = containerRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const time = (x / canvasWidth) * duration
    onSeek(Math.max(0, Math.min(duration, time)))
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-2">
      <div 
        ref={containerRef}
        className="relative cursor-pointer"
        onMouseMove={handleMouseMove}
        onMouseEnter={() => setIsHovering(true)}
        onMouseLeave={() => setIsHovering(false)}
        onClick={handleClick}
      >
        <canvas
          ref={canvasRef}
          width={canvasWidth}
          height={120}
          className="w-full border border-gray-200 rounded-lg bg-gray-50"
        />
        
        {/* Time tooltip */}
        {isHovering && (
          <div 
            className="absolute top-0 bg-gray-800 text-white px-2 py-1 rounded text-xs pointer-events-none transform -translate-x-1/2"
            style={{ left: `${(hoverTime / duration) * 100}%` }}
          >
            {formatTime(hoverTime)}
          </div>
        )}
      </div>

      {/* Tango Structure Visualization */}
      {tangoSections.length > 0 && (
        <div className="relative h-8 bg-gray-100 rounded-lg overflow-hidden">
          {tangoSections.map((section, index) => {
            const startPercent = (section.start / duration) * 100
            const widthPercent = ((section.end - section.start) / duration) * 100
            
            return (
              <div
                key={index}
                className={`absolute h-full flex items-center justify-center text-xs font-medium text-white ${
                  section.type === 'A' ? 'bg-blue-500' : 'bg-purple-500'
                }`}
                style={{
                  left: `${startPercent}%`,
                  width: `${widthPercent}%`
                }}
              >
                {section.type}
              </div>
            )
          })}
          
          {/* Current position indicator */}
          <div 
            className="absolute top-0 w-0.5 h-full bg-gray-800 pointer-events-none"
            style={{ left: `${(currentTime / duration) * 100}%` }}
          />
        </div>
      )}

      {/* Time display */}
      <div className="flex justify-between text-sm text-gray-500">
        <span>{formatTime(currentTime)}</span>
        <span>{formatTime(duration)}</span>
      </div>
    </div>
  )
}

export default MusicVisualizer

