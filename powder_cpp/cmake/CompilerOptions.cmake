function(powder_apply_strict_warnings target_name)
  if(MSVC)
    target_compile_options(${target_name} INTERFACE /W4 /WX)
  else()
    target_compile_options(${target_name} INTERFACE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Werror
    )
  endif()
endfunction()

function(powder_apply_profile_flags target_name)
  if(MSVC)
    return()
  endif()

  target_compile_options(${target_name} INTERFACE
    $<$<CONFIG:ReleaseLTO>:-O3>
    $<$<CONFIG:ReleaseLTO>:-flto>
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-O2>
    $<$<CONFIG:ASanUBSan>:-O1>
    $<$<CONFIG:ASanUBSan>:-g>
    $<$<CONFIG:ASanUBSan>:-fno-omit-frame-pointer>
    $<$<CONFIG:ASanUBSan>:-fsanitize=address,undefined>
  )

  target_link_options(${target_name} INTERFACE
    $<$<CONFIG:ReleaseLTO>:-flto>
    $<$<CONFIG:ASanUBSan>:-fsanitize=address,undefined>
  )
endfunction()
