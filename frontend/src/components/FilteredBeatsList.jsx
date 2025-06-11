import React from 'react'
import { Badge } from '@/components/ui/badge.jsx'
import { Button } from '@/components/ui/button.jsx'
import { Progress } from '@/components/ui/progress.jsx'

const FilteredBeatsList = ({ beats = [], currentTime, onBeatClick, confidenceThreshold = 0, strengthThreshold = 0 }) => {
  // Filter beats based on thresholds
  const filteredBeats = beats.filter(beat => {
    const confidence = (beat.confidence || 0) * 100
    const strength = (beat.strength || beat.confidence || 0) * 100
    return confidence >= confidenceThreshold && strength >= strengthThreshold
  })

  // Find current beat (closest to current time)
  const getCurrentBeatIndex = () => {
    let closestIndex = -1
    let closestDistance = Infinity
    
    filteredBeats.forEach((beat, index) => {
      const distance = Math.abs(beat.time - currentTime)
      if (distance < closestDistance) {
        closestDistance = distance
        closestIndex = index
      }
    })
    
    return closestIndex
  }

  const currentBeatIndex = getCurrentBeatIndex()

  // Get contextual beats (around current position)
  const getContextualBeats = () => {
    if (currentBeatIndex === -1) return filteredBeats.slice(0, 10)
    
    const start = Math.max(0, currentBeatIndex - 5)
    const end = Math.min(filteredBeats.length, currentBeatIndex + 6)
    return filteredBeats.slice(start, end)
  }

  const contextualBeats = getContextualBeats()

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const ms = Math.floor((seconds % 1) * 100)
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`
  }

  const getOriginalBeatNumber = (beat) => {
    return beats.indexOf(beat) + 1
  }

  const getFilteredBeatNumber = (beat) => {
    return filteredBeats.indexOf(beat) + 1
  }

  return (
    <div className="space-y-4">
      {/* Filter Summary */}
      <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
        <div className="text-sm">
          <span className="font-medium text-green-600">{filteredBeats.length}</span> of{' '}
          <span className="font-medium">{beats.length}</span> beats pass filters
        </div>
        <div className="flex gap-2">
          <Badge variant="outline" className="text-xs">
            Conf ≥ {confidenceThreshold}%
          </Badge>
          <Badge variant="outline" className="text-xs">
            Str ≥ {strengthThreshold}%
          </Badge>
        </div>
      </div>

      {/* Current Context */}
      {currentBeatIndex !== -1 && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="text-sm font-medium text-blue-800 mb-2">Current Context</h4>
          <div className="text-sm text-blue-700">
            Beat {getFilteredBeatNumber(filteredBeats[currentBeatIndex])} of {filteredBeats.length} filtered beats
            <br />
            (Original beat #{getOriginalBeatNumber(filteredBeats[currentBeatIndex])} of {beats.length} total)
          </div>
        </div>
      )}

      {/* Contextual Beats List */}
      <div className="space-y-2">
        <h4 className="font-medium text-sm flex items-center justify-between">
          <span>Beats Around Current Position</span>
          <Badge variant="secondary" className="text-xs">
            {contextualBeats.length} shown
          </Badge>
        </h4>
        
        <div className="max-h-64 overflow-y-auto space-y-1">
          {contextualBeats.map((beat, idx) => {
            const originalIndex = getOriginalBeatNumber(beat)
            const filteredIndex = getFilteredBeatNumber(beat)
            const isCurrentBeat = filteredBeats.indexOf(beat) === currentBeatIndex
            const confidence = (beat.confidence || 0) * 100
            const strength = (beat.strength || beat.confidence || 0) * 100
            
            return (
              <Button
                key={`${beat.time}-${idx}`}
                variant={isCurrentBeat ? "default" : "ghost"}
                className={`w-full justify-between p-3 h-auto ${
                  isCurrentBeat ? 'ring-2 ring-blue-500 bg-blue-100' : 'hover:bg-gray-50'
                }`}
                onClick={() => onBeatClick(beat.time)}
              >
                <div className="flex items-center space-x-3">
                  <div className="text-left">
                    <div className="font-medium text-sm">
                      Beat {filteredIndex}
                      <span className="text-xs text-gray-500 ml-1">
                        (#{originalIndex})
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">
                      {formatTime(beat.time)}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  {/* Confidence */}
                  <div className="text-right">
                    <div className="text-xs text-gray-500">Confidence</div>
                    <Badge 
                      variant={confidence >= 80 ? "default" : confidence >= 60 ? "secondary" : "outline"}
                      className="text-xs"
                    >
                      {confidence.toFixed(0)}%
                    </Badge>
                  </div>
                  
                  {/* Strength */}
                  <div className="text-right">
                    <div className="text-xs text-gray-500">Strength</div>
                    <Badge 
                      variant={strength >= 80 ? "default" : strength >= 60 ? "secondary" : "outline"}
                      className="text-xs"
                    >
                      {strength.toFixed(0)}%
                    </Badge>
                  </div>
                  
                  {/* Visual confidence bar */}
                  <div className="w-16">
                    <Progress value={confidence} className="h-2" />
                  </div>
                </div>
              </Button>
            )
          })}
        </div>
      </div>

      {/* Full Filtered List Toggle */}
      {filteredBeats.length > contextualBeats.length && (
        <details className="border border-gray-200 rounded-lg">
          <summary className="p-3 cursor-pointer hover:bg-gray-50 font-medium text-sm">
            Show All {filteredBeats.length} Filtered Beats
          </summary>
          <div className="p-3 border-t border-gray-200 max-h-96 overflow-y-auto space-y-1">
            {filteredBeats.map((beat, idx) => {
              const originalIndex = getOriginalBeatNumber(beat)
              const filteredIndex = idx + 1
              const isCurrentBeat = idx === currentBeatIndex
              const confidence = (beat.confidence || 0) * 100
              const strength = (beat.strength || beat.confidence || 0) * 100
              
              return (
                <Button
                  key={`${beat.time}-${idx}-full`}
                  variant={isCurrentBeat ? "default" : "ghost"}
                  className={`w-full justify-between p-2 h-auto text-xs ${
                    isCurrentBeat ? 'ring-1 ring-blue-500' : ''
                  }`}
                  onClick={() => onBeatClick(beat.time)}
                >
                  <div className="flex items-center space-x-2">
                    <span className="font-medium">
                      {filteredIndex}
                      <span className="text-gray-500 ml-1">(#{originalIndex})</span>
                    </span>
                    <span className="text-gray-500">{formatTime(beat.time)}</span>
                  </div>
                  
                  <div className="flex items-center space-x-1">
                    <Badge variant="outline" className="text-xs px-1">
                      C:{confidence.toFixed(0)}%
                    </Badge>
                    <Badge variant="outline" className="text-xs px-1">
                      S:{strength.toFixed(0)}%
                    </Badge>
                  </div>
                </Button>
              )
            })}
          </div>
        </details>
      )}

      {/* Statistics */}
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="font-medium text-gray-700">Average Confidence</div>
          <div className="text-lg font-bold text-green-600">
            {filteredBeats.length > 0 
              ? (filteredBeats.reduce((sum, beat) => sum + (beat.confidence || 0), 0) / filteredBeats.length * 100).toFixed(1)
              : 0
            }%
          </div>
        </div>
        
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="font-medium text-gray-700">Average Strength</div>
          <div className="text-lg font-bold text-blue-600">
            {filteredBeats.length > 0 
              ? (filteredBeats.reduce((sum, beat) => sum + (beat.strength || beat.confidence || 0), 0) / filteredBeats.length * 100).toFixed(1)
              : 0
            }%
          </div>
        </div>
      </div>
    </div>
  )
}

export default FilteredBeatsList

