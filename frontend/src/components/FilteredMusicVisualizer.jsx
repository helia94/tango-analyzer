import React, { useRef, useEffect, useState } from 'react'
import { ZoomIn, ZoomOut, RotateCcw, Filter, Sliders } from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Slider } from '@/components/ui/slider.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.jsx'

const FilteredMusicVisualizer = ({
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
  const [zoomLevel, setZoomLevel] = useState(16) // seconds to show (default 16s = 8s past + 8s future)

  // Filter states
  const [confidenceThreshold, setConfidenceThreshold] = useState([0]) // 0-100
  const [strengthThreshold, setStrengthThreshold] = useState([0]) // 0-100
  const [showFilters, setShowFilters] = useState(false)

  // Zoom level options
  const zoomOptions = [
    { value: 8, label: '8s', description: '4s past + 4s future' },
    { value: 16, label: '16s', description: '8s past + 8s future' },
    { value: 32, label: '32s', description: '16s past + 16s future' },
    { value: 60, label: '1min', description: '30s past + 30s future' },
    { value: 'full', label: 'Full', description: 'Entire song' }
  ]

  // Filter beats based on thresholds
  const getFilteredBeats = () => {
    return beats.filter(beat => {
      const confidence = (beat.confidence || 0) * 100
      const strength = (beat.strength || beat.confidence || 0) * 100
      return confidence >= confidenceThreshold[0] && strength >= strengthThreshold[0]
    })
  }

  const filteredBeats = getFilteredBeats()

  // Calculate visible time window
  const getVisibleTimeWindow = () => {
    if (zoomLevel === 'full') {
      return { start: 0, end: duration }
    }

    const halfZoom = zoomLevel / 2
    const start = Math.max(0, currentTime - halfZoom)
    const end = Math.min(duration, currentTime + halfZoom)

    // If we're near the beginning or end, adjust the window
    if (start === 0) {
      return { start: 0, end: Math.min(duration, zoomLevel) }
    }
    if (end === duration) {
      return { start: Math.max(0, duration - zoomLevel), end: duration }
    }

    return { start, end }
  }

  const { start: windowStart, end: windowEnd } = getVisibleTimeWindow()
  const windowDuration = windowEnd - windowStart

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
    if (!canvas || duration === 0 || windowDuration === 0) return

    const ctx = canvas.getContext('2d')
    const height = canvas.height
    const width = canvas.width

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw background
    ctx.fillStyle = '#f8fafc'
    ctx.fillRect(0, 0, width, height)

    // Draw time grid lines (every second for detailed view)
    ctx.strokeStyle = '#e2e8f0'
    ctx.lineWidth = 1
    const secondsInView = Math.ceil(windowDuration)
    for (let i = 0; i <= secondsInView; i++) {
      const time = Math.floor(windowStart) + i
      if (time <= windowEnd) {
        const x = ((time - windowStart) / windowDuration) * width
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, height - 30)
        ctx.stroke()

        // Time labels
        ctx.fillStyle = '#64748b'
        ctx.font = '10px sans-serif'
        ctx.textAlign = 'center'
        const mins = Math.floor(time / 60)
        const secs = time % 60
        ctx.fillText(`${mins}:${secs.toString().padStart(2, '0')}`, x, height - 15)
      }
    }

    // Draw progress background
    ctx.fillStyle = '#e2e8f0'
    ctx.fillRect(0, height - 8, width, 8)

    // Draw progress (only the visible portion)
    if (currentTime >= windowStart && currentTime <= windowEnd) {
      const progressStart = Math.max(0, (Math.max(windowStart, 0) - windowStart) / windowDuration * width)
      const progressEnd = Math.min(width, (currentTime - windowStart) / windowDuration * width)
      ctx.fillStyle = '#3b82f6'
      ctx.fillRect(progressStart, height - 8, progressEnd - progressStart, 8)
    }

    if (visualizationMode === 'beats') {
      // Draw filtered beats in visible window
      filteredBeats.forEach((beat, index) => {
        if (beat.time >= windowStart && beat.time <= windowEnd) {
          const x = ((beat.time - windowStart) / windowDuration) * width
          const intensity = beat.confidence || 0.5
          const strength = beat.strength || beat.confidence || 0.5

          // Beat line - stronger beats get thicker lines
          ctx.strokeStyle = `rgba(239, 68, 68, ${0.4 + intensity * 0.6})`
          ctx.lineWidth = 2 + strength * 3
          ctx.beginPath()
          ctx.moveTo(x, 20)
          ctx.lineTo(x, height - 40)
          ctx.stroke()

          // Beat marker - size based on strength
          ctx.fillStyle = `rgba(239, 68, 68, ${0.6 + intensity * 0.4})`
          ctx.beginPath()
          ctx.arc(x, height / 2, 3 + strength * 6, 0, 2 * Math.PI)
          ctx.fill()

          // Beat number and confidence
          ctx.fillStyle = '#1f2937'
          ctx.font = '11px sans-serif'
          ctx.textAlign = 'center'
          const originalBeatIndex = beats.indexOf(beat) + 1
          ctx.fillText(`${originalBeatIndex}`, x, 15)

          ctx.font = '9px sans-serif'
          ctx.fillStyle = '#6b7280'
          ctx.fillText(`${Math.round(intensity * 100)}%`, x, height - 25)

          // Strength indicator
          ctx.font = '8px sans-serif'
          ctx.fillStyle = '#059669'
          ctx.fillText(`S:${Math.round(strength * 100)}`, x, height - 35)
        }
      })

      // Draw filtered out beats as faded markers (optional - to show what's being filtered)
      if (showFilters) {
        beats.forEach(beat => {
          if (!filteredBeats.includes(beat) && beat.time >= windowStart && beat.time <= windowEnd) {
            const x = ((beat.time - windowStart) / windowDuration) * width

            // Faded beat line
            ctx.strokeStyle = 'rgba(156, 163, 175, 0.3)'
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(x, height / 2 - 10)
            ctx.lineTo(x, height / 2 + 10)
            ctx.stroke()

            // Small faded marker
            ctx.fillStyle = 'rgba(156, 163, 175, 0.4)'
            ctx.beginPath()
            ctx.arc(x, height / 2, 2, 0, 2 * Math.PI)
            ctx.fill()
          }
        })
      }
    } else if (visualizationMode === 'melody') {
      // Draw melody segments in visible window
      const centerY = height / 2

      melodySegments.forEach((segment, index) => {
        // Check if segment overlaps with visible window
        if (segment.end >= windowStart && segment.start <= windowEnd) {
          const startX = Math.max(0, ((segment.start - windowStart) / windowDuration) * width)
          const endX = Math.min(width, ((segment.end - windowStart) / windowDuration) * width)
          const segmentWidth = endX - startX

          if (segmentWidth < 1) return

          const intensity = segment.confidence || 0.5

          if (segment.type === 'staccato') {
            // Spiky line for staccato - more detailed
            ctx.strokeStyle = `rgba(168, 85, 247, ${0.5 + intensity * 0.5})`
            ctx.lineWidth = 2
            ctx.beginPath()

            const spikes = Math.max(5, Math.floor(segmentWidth / 5)) // More spikes for detail
            for (let i = 0; i <= spikes; i++) {
              const x = startX + (i / spikes) * segmentWidth
              const amplitude = 20 + intensity * 20
              const y = i % 2 === 0 ? centerY - amplitude : centerY + amplitude
              if (i === 0) ctx.moveTo(x, y)
              else ctx.lineTo(x, y)
            }
            ctx.stroke()

            // Add staccato markers
            for (let i = 0; i <= spikes; i += 2) {
              const x = startX + (i / spikes) * segmentWidth
              ctx.fillStyle = `rgba(168, 85, 247, ${0.7 + intensity * 0.3})`
              ctx.beginPath()
              ctx.arc(x, centerY - (20 + intensity * 20), 2, 0, 2 * Math.PI)
              ctx.fill()
            }
          } else {
            // Smooth line for legato - more detailed
            ctx.strokeStyle = `rgba(34, 197, 94, ${0.5 + intensity * 0.5})`
            ctx.lineWidth = 3
            ctx.beginPath()

            const amplitude = 15 + intensity * 20
            const frequency = 0.1 // Higher frequency for more detail

            for (let x = startX; x <= endX; x += 1) {
              const progress = (x - startX) / segmentWidth
              const y = centerY + Math.sin(progress * Math.PI * 4 + segment.start) * amplitude
              if (x === startX) ctx.moveTo(x, y)
              else ctx.lineTo(x, y)
            }
            ctx.stroke()
          }

          // Segment background
          ctx.fillStyle = segment.type === 'staccato'
            ? `rgba(168, 85, 247, 0.1)`
            : `rgba(34, 197, 94, 0.1)`
          ctx.fillRect(startX, 20, segmentWidth, height - 50)

          // Segment label
          if (segmentWidth > 30) {
            ctx.fillStyle = segment.type === 'staccato' ? '#7c3aed' : '#059669'
            ctx.font = '10px sans-serif'
            ctx.textAlign = 'center'
            ctx.fillText(segment.type.toUpperCase(), startX + segmentWidth / 2, 35)
          }
        }
      })
    }

    // Draw current time indicator
    if (currentTime >= windowStart && currentTime <= windowEnd) {
      const currentX = ((currentTime - windowStart) / windowDuration) * width
      ctx.strokeStyle = '#1f2937'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(currentX, 0)
      ctx.lineTo(currentX, height - 8)
      ctx.stroke()

      // Current time marker
      ctx.fillStyle = '#1f2937'
      ctx.beginPath()
      ctx.arc(currentX, height / 2, 6, 0, 2 * Math.PI)
      ctx.fill()
    }

    // Draw hover indicator
    if (isHovering && hoverTime >= windowStart && hoverTime <= windowEnd) {
      const hoverX = ((hoverTime - windowStart) / windowDuration) * width
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.7)'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.moveTo(hoverX, 0)
      ctx.lineTo(hoverX, height - 8)
      ctx.stroke()
      ctx.setLineDash([])
    }

  }, [duration, currentTime, filteredBeats, melodySegments, visualizationMode, canvasWidth, isHovering, hoverTime, windowStart, windowEnd, windowDuration, zoomLevel, showFilters, beats])

  const handleMouseMove = (e) => {
    if (!containerRef.current || windowDuration === 0) return

    const rect = containerRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const time = windowStart + (x / canvasWidth) * windowDuration
    setHoverTime(Math.max(0, Math.min(duration, time)))
  }

  const handleClick = (e) => {
    if (!containerRef.current || windowDuration === 0) return

    const rect = containerRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const time = windowStart + (x / canvasWidth) * windowDuration
    onSeek(Math.max(0, Math.min(duration, time)))
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const handleZoomChange = (newZoom) => {
    setZoomLevel(newZoom)
  }

  const resetFilters = () => {
    setConfidenceThreshold([0])
    setStrengthThreshold([0])
  }

  return (
    <div className="space-y-4">
      {/* Controls Row */}
      <div className="flex flex-col lg:flex-row gap-4">
        {/* Zoom Controls */}
        <div className="flex items-center justify-between bg-gray-50 p-3 rounded-lg flex-1">
          <div className="flex items-center space-x-2">
            <ZoomIn className="h-4 w-4 text-gray-600" />
            <span className="text-sm font-medium text-gray-700">Zoom:</span>
          </div>

          <div className="flex items-center space-x-2">
            {zoomOptions.map((option) => (
              <Button
                key={option.value}
                variant={zoomLevel === option.value ? "default" : "outline"}
                size="sm"
                onClick={() => handleZoomChange(option.value)}
                className="text-xs"
              >
                {option.label}
              </Button>
            ))}
          </div>
        </div>

        {/* Filter Toggle */}
        <div className="flex items-center space-x-2">
          <Button
            variant={showFilters ? "default" : "outline"}
            size="sm"
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center gap-2"
          >
            <Filter className="h-4 w-4" />
            Filters
          </Button>
        </div>
      </div>

      {/* Filter Controls */}
      {showFilters && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <Sliders className="h-4 w-4" />
              Beat Filters
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Confidence Threshold */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label className="text-sm font-medium">Confidence Threshold</Label>
                  <Badge variant="secondary" className="text-xs">
                    {confidenceThreshold[0]}%
                  </Badge>
                </div>
                <Slider
                  value={confidenceThreshold}
                  onValueChange={setConfidenceThreshold}
                  max={100}
                  min={0}
                  step={5}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>0% (All beats)</span>
                  <span>100% (Only certain beats)</span>
                </div>
              </div>

              {/* Strength Threshold */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label className="text-sm font-medium">Strength Threshold</Label>
                  <Badge variant="secondary" className="text-xs">
                    {strengthThreshold[0]}%
                  </Badge>
                </div>
                <Slider
                  value={strengthThreshold}
                  onValueChange={setStrengthThreshold}
                  max={100}
                  min={0}
                  step={5}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>0% (All beats)</span>
                  <span>100% (Only strong beats)</span>
                </div>
              </div>
            </div>

            {/* Filter Stats and Reset */}
            <div className="flex justify-between items-center pt-2 border-t">
              <div className="text-sm text-gray-600">
                <span className="font-medium text-green-600">{filteredBeats.length}</span> of{' '}
                <span className="font-medium">{beats.length}</span> beats shown
                {filteredBeats.length < beats.length && (
                  <span className="text-red-600 ml-2">
                    ({beats.length - filteredBeats.length} filtered out)
                  </span>
                )}
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={resetFilters}
                className="flex items-center gap-2"
              >
                <RotateCcw className="h-3 w-3" />
                Reset
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Visualization */}
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
          height={200}
          className="w-full border border-gray-200 rounded-lg bg-gray-50"
        />

        {/* Time tooltip */}
        {isHovering && hoverTime >= windowStart && hoverTime <= windowEnd && (
          <div
            className="absolute top-0 bg-gray-800 text-white px-2 py-1 rounded text-xs pointer-events-none transform -translate-x-1/2 z-10"
            style={{ left: `${((hoverTime - windowStart) / windowDuration) * 100}%` }}
          >
            {formatTime(hoverTime)}
          </div>
        )}
      </div>

      {/* Tango Structure Visualization */}
      {tangoSections.length > 0 && (
        <div className="relative h-10 bg-gray-100 rounded-lg overflow-hidden">
          {tangoSections.map((section, index) => {
            // Only show sections that overlap with visible window
            if (section.end >= windowStart && section.start <= windowEnd) {
              const startPercent = Math.max(0, ((section.start - windowStart) / windowDuration) * 100)
              const endPercent = Math.min(100, ((section.end - windowStart) / windowDuration) * 100)
              const widthPercent = endPercent - startPercent

              return (
                <div
                  key={index}
                  className={`absolute h-full flex items-center justify-center text-sm font-bold text-white ${section.type === 'A' ? 'bg-blue-500' : 'bg-purple-500'
                    }`}
                  style={{
                    left: `${startPercent}%`,
                    width: `${widthPercent}%`
                  }}
                >
                  {widthPercent > 10 && section.type}
                </div>
              )
            }
            return null
          })}

          {/* Current position indicator */}
          {currentTime >= windowStart && currentTime <= windowEnd && (
            <div
              className="absolute top-0 w-1 h-full bg-gray-800 pointer-events-none z-10"
              style={{ left: `${((currentTime - windowStart) / windowDuration) * 100}%` }}
            />
          )}
        </div>
      )}

      {/* Window info */}
      <div className="flex justify-between text-sm text-gray-500 bg-gray-50 p-2 rounded">
        <span>Window: {formatTime(windowStart)} - {formatTime(windowEnd)}</span>
        <span>Current: {formatTime(currentTime)}</span>
        <span>
          Showing: {filteredBeats.length}/{beats.length} beats
        </span>
      </div>
    </div>
  )
}

export default FilteredMusicVisualizer

